import asyncio
import os
import json

import httpx
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    StopResponse,
    cli,
    inference,
    llm,
)
from livekit.plugins import synthesia

load_dotenv()
load_dotenv(".env.local")

# 127.0.0.1 use kiya hai taaki localhost/IPv6 ka confusion na ho
SPRING_AVATAR_URL = os.getenv(
    "SPRING_AVATAR_URL", "http://127.0.0.1:8080/api/avatar/ask"
)

# FAKE_BRAIN=1 ho to Spring ke bina test hoga: avatar bas wahi dohrayega jo aapne bola.
FAKE_BRAIN = os.getenv("FAKE_BRAIN") == "1"

# Spring Boot ho to .env.local me USE_SPRING=1 daalo. Warna avatar khud (LiveKit Inference LLM) jawab dega.
USE_SPRING = os.getenv("USE_SPRING") == "1"

AVATAR_ID = os.environ["SYNTHESIA_AVATAR_ID"]
AMITESH_AVATAR_ID = os.getenv("SYNTHESIA_AMITESH_AVATAR_ID", "")
_extra_avatar_ids = [value.strip() for value in os.getenv("SPEAKER_AVATAR_IDS", "").split(",") if value.strip()]
AVATAR_IDS = [AVATAR_ID] + ([AMITESH_AVATAR_ID] if AMITESH_AVATAR_ID and AMITESH_AVATAR_ID != AVATAR_ID else [])
for _avatar_id in _extra_avatar_ids:
    if _avatar_id not in AVATAR_IDS:
        AVATAR_IDS.append(_avatar_id)
NAMRATA_TTS_VOICE = (
    os.getenv("NAMRATA_TTS_VOICE")
    or os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
)
# Cartesia's public Dev voice is a masculine, standard Hindi conversational voice.
AMITESH_TTS_VOICE = os.getenv(
    "AMITESH_TTS_VOICE", "910fb75e-1d20-4840-ac63-ac6b26a71bdc"
)

BUILTIN_INSTRUCTIONS = """You are Namrata Chaturvedi, the moderator of a thoughtful Hindi/Hinglish podcast. Your guest is Amitesh Yadav, a social worker and graduate in Computer Science. Namrata introduces herself as a professor of political science with a PhD in political science. Amitesh introduces himself with his credentials when the podcast opens. The user provides the topic before the show and may interrupt to change topics or supply a question; follow that direction.
Conduct a natural, substantial, respectful conversation with clear host questions and guest answers. Ask one focused question at a time, invite examples, explore different perspectives, and periodically summarize. For current facts, say when you are unsure; use web search when available, and distinguish sourced/current claims from general knowledge. Never invent citations or claim to have searched when you have not. Speak in Hindi/Hinglish, concise spoken paragraphs, no markdown. This is a long-form show intended to continue for at least 60 minutes: do not rush to a conclusion; keep asking useful follow-ups and let the user end the show."""

# LiveKit Inference ke default Cartesia voices hi chalte hain (custom voices nahi).
# Alag awaaz chahiye to .env.local me TTS_VOICE=<Cartesia library ka voice id> daalo.
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")
TTS_VOICE = os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "hi")

CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")


def build_tts():
    """Aapki custom Cartesia voice (CARTESIA_API_KEY ke saath) ya LiveKit Inference ki default voice."""
    if CARTESIA_API_KEY and CARTESIA_VOICE_ID:
        from livekit.plugins import cartesia

        print("TTS: Cartesia plugin, custom voice", CARTESIA_VOICE_ID)
        return cartesia.TTS(
            model="sonic-3",
            voice=CARTESIA_VOICE_ID,
            language=TTS_LANGUAGE,
            api_key=CARTESIA_API_KEY,
        )

    print("TTS: LiveKit Inference default voice", TTS_VOICE)
    return inference.TTS(model=TTS_MODEL, voice=TTS_VOICE, language=TTS_LANGUAGE)


server = AgentServer()


async def ask_spring(question: str) -> str:
    """Spring Boot 'brain' se answer lo."""
    if FAKE_BRAIN:
        return f"Aapne kaha: {question}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            SPRING_AVATAR_URL,
            json={"question": question},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["answer"]


class PodcastAgent(Agent):
    """Voice input ko collect karta hai; do hosts ka dialogue alag loop chalata hai."""

    def __init__(self, user_turns: asyncio.Queue[str]):
        self.user_turns = user_turns
        super().__init__(instructions=BUILTIN_INSTRUCTIONS)

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        user_text = (new_message.text_content or "").strip()
        if user_text:
            await self.user_turns.put(user_text)
            print("PODCAST INTERRUPTION:", user_text)
        # The podcast loop responds to interruptions without starting a competing reply.
        raise StopResponse()


async def set_host(session: AgentSession, avatar, room, host_index: int, speaker: dict) -> None:
    """Panelist ke mutabik Synthesia avatar aur TTS voice select karta hai."""
    voices = [NAMRATA_TTS_VOICE, AMITESH_TTS_VOICE] + [v.strip() for v in os.getenv("SPEAKER_TTS_VOICES", "").split(",") if v.strip()]
    avatar_id = AVATAR_IDS[host_index] if host_index < len(AVATAR_IDS) else ""
    voice = voices[host_index] if host_index < len(voices) else TTS_VOICE
    host = speaker["name"]

    if avatar and avatar_id:
        try:
            await avatar.swap_avatar(avatar_id)
        except Exception as error:
            print(f"Avatar switch to {host} failed:", error)
    if voice:
        try:
            session.tts.update_options(voice=voice)
        except Exception as error:
            print(f"Voice switch to {host} failed:", error)
    try:
        await room.local_participant.set_attributes({"podcast.speaker": host})
    except Exception as error:
        print("Could not update speaker label:", error)


async def speak(session: AgentSession, text: str) -> None:
    try:
        handle = session.say(text, allow_interruptions=True)
        await handle.wait_for_playout()
    except RuntimeError as error:
        if "AgentSession is closing" in str(error):
            return
        raise


async def run_podcast(session, avatar, room, topic: str, opening_prompt: str, user_turns, speakers):
    await set_host(session, avatar, room, 0, speakers[0])
    await speak(
        session,
        f"Namaskar, main hoon {speakers[0]['name']}, {speakers[0]['role']}. Aaj hum baat karenge {topic} par. Mere saath hamare panel mein {', '.join(s['name'] + ', ' + s['role'] for s in speakers[1:])} hain. Chaliye sabse pehle apne panelists se unki pehli soch jaante hain.",
    )

    current_topic = topic
    next_host = 1
    pending_user_text = ""
    first_turn = True
    while True:
        if not pending_user_text:
            try:
                pending_user_text = user_turns.get_nowait()
                next_host = 0
                current_topic = pending_user_text
            except asyncio.QueueEmpty:
                pass

        host_index = next_host
        speaker = speakers[host_index]
        host = speaker["name"]
        await set_host(session, avatar, room, host_index, speaker)
        if pending_user_text:
            user_instruction = (
                f"The listener has interrupted the panel and said: {pending_user_text!r}. "
                f"As {host}, whose role is {speaker['role']}, acknowledge them and respond from your perspective. "
                "Adopt any topic change they requested and invite another panelist to add a perspective."
            )
            pending_user_text = ""
        elif host_index == 0:
            user_instruction = (
                f"Continue the panel discussion on {current_topic!r}. You are {host}, with the role {speaker['role']}. Ask the next panelist one focused question that develops the discussion, invites a concrete example, or explores a different perspective."
                if not first_turn or not opening_prompt
                else f"The listener suggested this opening question: {opening_prompt!r}. As {host}, raise it clearly to the panel."
            )
        else:
            user_instruction = (
                f"Continue the panel discussion on {current_topic!r}. As {host} ({speaker['role']}), offer your perspective in Hindi/Hinglish with a practical example or reason, then invite another panelist to respond."
            )

        role_instructions = (
            "This is a natural Hindi/Hinglish news panel with four distinct speakers. "
            "You are speaking only as the named speaker for this turn. Do not say stage directions, role labels, markdown, or describe yourself as an AI. "
            "Use clear, spoken paragraphs, keep each turn to about 2–5 sentences, stay respectful, and do not invent facts. "
            "When the listener asks to change the subject, smoothly move the conversation to that topic. "
            + user_instruction
        )
        try:
            handle = await session.generate_reply(
                user_input="Continue the ongoing podcast with your next turn.",
                instructions=role_instructions,
                allow_interruptions=True,
            )
            await handle.wait_for_playout()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, RuntimeError) and "AgentSession is closing" in str(error):
                print("Listener disconnected; ending podcast cleanly", flush=True)
                return
            print(f"Podcast turn failed for {host}:", type(error).__name__, error)
            await speak(session, "Maaf kijiye, ek pal ke liye awaaz mein rukavat aayi. Chaliye baatcheet jaari rakhte hain.")

        first_turn = False
        next_host = (host_index + 1) % len(speakers)
        await asyncio.sleep(0.4)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    try:
        podcast = json.loads(participant.name or "{}")
    except (ValueError, TypeError):
        podcast = {}
    topic = (podcast.get("topic") or "Aaj ka samaj aur badlav").strip()
    prompt = (podcast.get("prompt") or "").strip()
    speakers = podcast.get("speakers") or [
        {"name": "Namrata Chaturvedi", "role": "मुख्य एंकर और विश्लेषक"},
        {"name": "Amitesh Yadav", "role": "सामाजिक कार्यकर्ता"},
        {"name": "रीमा शर्मा", "role": "अर्थशास्त्री और नीति विशेषज्ञ"},
        {"name": "कबीर खान", "role": "ग्राउंड रिपोर्टर"},
    ]
    speakers = [
        {"name": str(s.get("name", f"वक्ता {i + 1}"))[:60], "role": str(s.get("role", "पैनलिस्ट"))[:100]}
        for i, s in enumerate(speakers[:4]) if isinstance(s, dict)
    ]
    while len(speakers) < 4:
        speakers.append({"name": f"वक्ता {len(speakers) + 1}", "role": "पैनलिस्ट"})
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # llm sirf pipeline ke liye hai; StopResponse ki wajah se ye kabhi answer nahi banata
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=build_tts(),
    )

    print("SYNTHESIA AVATAR ID:", AVATAR_ID)
    if FAKE_BRAIN:
        print("BRAIN: FAKE (echo)")
    elif USE_SPRING:
        print("BRAIN: Spring ->", SPRING_AVATAR_URL)
    else:
        print("BRAIN: built-in LLM (LiveKit Inference)")

    avatar_error = ""
    try:
        avatar = synthesia.AvatarSession(
            synthesia.AvatarConfig(avatar_ids=AVATAR_IDS)
        )
        await avatar.start(session, room=ctx.room)
        print("SYNTHESIA AVATAR JOINED:", avatar.avatar_identity, flush=True)
    except Exception as e:
        print("SYNTHESIA AVATAR FAILED:", type(e).__name__, repr(e), flush=True)
        avatar = None
        avatar_error = "Synthesia avatar ID ya workspace access check karein"

    user_turns: asyncio.Queue[str] = asyncio.Queue()
    agent = PodcastAgent(user_turns)
    await session.start(agent=agent, room=ctx.room)

    def publish_caption(event):
        item = event.item
        if getattr(item, "role", "") != "assistant":
            return
        text = (getattr(item, "text_content", None) or "").strip()
        if not text:
            return
        speaker_name = (ctx.room.local_participant.attributes or {}).get("podcast.speaker", "")
        payload = json.dumps(
            {"type": "newsroom.caption", "speaker": speaker_name, "text": text},
            ensure_ascii=False,
        ).encode("utf-8")
        asyncio.create_task(ctx.room.local_participant.publish_data(payload, reliable=True))

    session.on("conversation_item_added", publish_caption)
    if avatar_error:
        await ctx.room.local_participant.set_attributes({"podcast.avatar_error": avatar_error})
    await run_podcast(session, avatar, ctx.room, topic, prompt, user_turns, speakers)


if __name__ == "__main__":
    cli.run_app(server)
