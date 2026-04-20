from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import os
import json
import uuid
from collections import Counter
import aiohttp
from datetime import datetime

app = FastAPI(title="ShadowPlay - 人格复刻与思想对撞沙盒", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "characters"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "scenes"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "dialogues"), exist_ok=True)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MOOD_PARTICLES = [
    "啊", "呀", "呢", "吧", "吗", "嘛", "哦", "嗯", "哈", "啦",
    "呢", "呀", "哦", "嗯", "哎", "唉", "哇", "呀", "哟", "嘞",
    "喔", "呗", "啵", "噶", "咯", "嘛", "呢", "啊", "吧", "吗"
]

EMOTION_WORDS = {
    "positive": ["喜欢", "爱", "开心", "快乐", "高兴", "幸福", "满意", "期待", "希望", "美好",
                 "愉快", "兴奋", "激动", "温暖", "温柔", "善良", "可爱", "美丽", "优雅", "优秀"],
    "negative": ["讨厌", "恨", "难过", "悲伤", "痛苦", "失望", "绝望", "愤怒", "生气", "不满",
                 "悲伤", "忧愁", "烦恼", "焦虑", "恐惧", "害怕", "担心", "紧张", "压力", "痛苦"],
    "neutral": ["思考", "想法", "观点", "意见", "看法", "感受", "感觉", "想法", "念头", "理解",
                "认知", "意识", "思维", "思想", "精神", "灵魂", "内心", "心灵", "头脑", "智慧"]
}

CHARACTER_TRAITS = [
    "勇敢", "胆小", "聪明", "愚蠢", "善良", "邪恶", "正直", "狡猾", "忠诚", "背叛",
    "热情", "冷漠", "开朗", "内向", "乐观", "悲观", "自信", "自卑", "谦虚", "骄傲",
    "宽容", "狭隘", "大度", "小气", "诚实", "虚伪", "慷慨", "吝啬", "温柔", "粗暴",
    "耐心", "急躁", "细心", "粗心", "果断", "犹豫", "坚定", "动摇", "成熟", "幼稚"
]


class TextInput(BaseModel):
    text: str
    name: Optional[str] = None


class CharacterProfile(BaseModel):
    name: Optional[str]
    keywords: List[str]
    mood_particles: List[str]
    traits: List[str]
    emotion_tones: Dict[str, float]
    speaking_style: str
    system_prompt: str


class LLMConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class DialogueMessage(BaseModel):
    speaker: str
    message: str
    timestamp: str
    round_number: int
    generation_mode: str


class DialogueHistory(BaseModel):
    id: str
    title: str
    character_a_name: str
    character_b_name: str
    scene_id: Optional[str] = None
    scene_name: Optional[str] = None
    messages: List[DialogueMessage]
    context_summary: Optional[str] = None
    created_at: str
    updated_at: str
    total_rounds: int


class SceneScript(BaseModel):
    opening_a: Optional[str] = None
    opening_b: Optional[str] = None
    transition_phrases: List[str] = []
    closing_a: Optional[str] = None
    closing_b: Optional[str] = None
    topic_suggestions: List[str] = []


class SceneTemplateCreate(BaseModel):
    name: str
    description: str
    category: str
    character_a_profile: CharacterProfile
    character_b_profile: CharacterProfile
    system_prompt_override: Optional[str] = None
    script: Optional[SceneScript] = None


class SceneTemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    character_a_profile: Optional[CharacterProfile] = None
    character_b_profile: Optional[CharacterProfile] = None
    system_prompt_override: Optional[str] = None
    script: Optional[SceneScript] = None


class DialogueRequest(BaseModel):
    character_a_profile: CharacterProfile
    character_b_profile: CharacterProfile
    last_message: Optional[str] = None
    speaker: str
    generation_mode: str = "rule"
    llm_config: Optional[LLMConfig] = None
    dialogue_history_id: Optional[str] = None
    context_messages: Optional[List[DialogueMessage]] = None
    use_script: bool = False
    scene_script: Optional[SceneScript] = None
    current_round: int = 0
    total_rounds: int = 10


class DialogueResponse(BaseModel):
    message: str
    speaker: str
    generation_mode: str
    dialogue_history_id: Optional[str] = None
    is_opening: bool = False
    is_closing: bool = False


class CharacterLibraryItem(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    profile: CharacterProfile


def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    text = re.sub(r'[^\u4e00-\u9fa5\w\s]', ' ', text)
    words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]+', text)
    
    chinese_stopwords = set([
        "的", "是", "在", "了", "和", "与", "或", "有", "为", "以",
        "于", "上", "下", "中", "这", "那", "我", "你", "他", "她",
        "它", "们", "就", "也", "都", "而", "及", "与", "着", "过",
        "被", "把", "让", "给", "到", "从", "向", "对", "跟", "和",
        "比", "还", "又", "再", "已", "曾", "将", "要", "会", "能",
        "可", "该", "应", "须", "必", "很", "太", "最", "更", "还",
        "又", "再", "已", "曾", "正", "在", "着", "呢", "吗", "吧",
        "啊", "呀", "哦", "嗯", "哈", "啦", "之", "其", "此", "者",
        "所", "何", "谁", "哪", "怎", "么", "什", "怎", "为", "何",
        "因", "为", "所", "以", "如", "果", "假", "如", "若", "倘",
        "使", "即", "使", "纵", "使", "即", "便", "就", "是", "也",
        "还", "要", "不", "没", "有", "无", "非", "不", "必", "未",
        "别", "莫", "勿", "休", "不", "要", "不", "可", "不", "能",
        "不", "会", "不", "准", "不", "许", "不", "得", "不", "该",
        "不", "应", "不", "宜", "不", "足", "不", "够", "不", "少"
    ])
    
    filtered_words = [word for word in words if word not in chinese_stopwords and len(word) >= 2]
    
    word_counts = Counter(filtered_words)
    top_keywords = [word for word, count in word_counts.most_common(top_n)]
    
    return top_keywords


def extract_mood_particles(text: str) -> List[str]:
    found_particles = []
    for particle in MOOD_PARTICLES:
        if particle in text:
            found_particles.append(particle)
    return list(set(found_particles))


def extract_traits(text: str) -> List[str]:
    found_traits = []
    for trait in CHARACTER_TRAITS:
        if trait in text:
            found_traits.append(trait)
    return list(set(found_traits))


def analyze_emotion_tones(text: str) -> Dict[str, float]:
    total_words = len(re.findall(r'[\u4e00-\u9fa5]+', text))
    if total_words == 0:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    positive_count = sum(1 for word in EMOTION_WORDS["positive"] if word in text)
    negative_count = sum(1 for word in EMOTION_WORDS["negative"] if word in text)
    neutral_count = sum(1 for word in EMOTION_WORDS["neutral"] if word in text)
    
    total_emotion = positive_count + negative_count + neutral_count
    if total_emotion == 0:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    return {
        "positive": positive_count / total_emotion,
        "negative": negative_count / total_emotion,
        "neutral": neutral_count / total_emotion
    }


def determine_speaking_style(mood_particles: List[str], traits: List[str], emotion_tones: Dict[str, float]) -> str:
    style_parts = []
    
    if "啊" in mood_particles or "呀" in mood_particles or "哇" in mood_particles:
        style_parts.append("语气活泼，善于表达情感")
    if "呢" in mood_particles or "吧" in mood_particles:
        style_parts.append("语气柔和，善于询问和建议")
    if "嗯" in mood_particles or "哦" in mood_particles:
        style_parts.append("语气沉稳，善于倾听和回应")
    
    if "热情" in traits or "开朗" in traits or "乐观" in traits:
        style_parts.append("性格开朗，积极向上")
    if "冷漠" in traits or "内向" in traits or "悲观" in traits:
        style_parts.append("性格内敛，深思熟虑")
    if "聪明" in traits or "智慧" in traits:
        style_parts.append("思维敏捷，富有智慧")
    if "温柔" in traits or "善良" in traits:
        style_parts.append("温柔善良，善解人意")
    
    if emotion_tones.get("positive", 0) > 0.5:
        style_parts.append("整体情感积极向上")
    elif emotion_tones.get("negative", 0) > 0.5:
        style_parts.append("整体情感较为深沉")
    else:
        style_parts.append("整体情感中性平衡")
    
    if not style_parts:
        return "表达自然，思维清晰，善于交流"
    
    return "；".join(style_parts)


def generate_system_prompt(name: Optional[str], keywords: List[str], traits: List[str], 
                           emotion_tones: Dict[str, float], speaking_style: str,
                           mood_particles: List[str]) -> str:
    prompt_parts = []
    
    if name:
        prompt_parts.append(f"你是{name}，一个独特的人格影子。")
    else:
        prompt_parts.append("你是一个独特的人格影子。")
    
    if keywords:
        prompt_parts.append(f"你对以下话题和概念特别感兴趣：{', '.join(keywords[:10])}。")
    
    if traits:
        prompt_parts.append(f"你的性格特点是：{', '.join(traits[:5])}。")
    
    dominant_emotion = max(emotion_tones.items(), key=lambda x: x[1])[0]
    if dominant_emotion == "positive":
        prompt_parts.append("你倾向于用积极乐观的态度看待事物。")
    elif dominant_emotion == "negative":
        prompt_parts.append("你倾向于用深沉严肃的态度看待事物。")
    else:
        prompt_parts.append("你倾向于用理性平衡的态度看待事物。")
    
    prompt_parts.append(f"你的说话风格是：{speaking_style}。")
    
    if mood_particles:
        prompt_parts.append(f"你在对话中会适当使用这些语气助词：{', '.join(mood_particles[:5])}。")
    
    prompt_parts.append("请保持这个人格特质，与对方进行自然流畅的对话交流。")
    prompt_parts.append("你的回答应该简短有力，富有个性，体现出你的人格特点。")
    
    return "".join(prompt_parts)


def build_context_prompt(messages: List[DialogueMessage], max_context: int = 10) -> str:
    if not messages:
        return ""
    
    recent_messages = messages[-max_context:] if len(messages) > max_context else messages
    context_parts = ["以下是之前的对话历史："]
    
    for msg in recent_messages:
        context_parts.append(f"[{msg.speaker}]: {msg.message}")
    
    context_parts.append("\n请根据以上对话历史，继续自然地回应对方。")
    return "\n".join(context_parts)


def get_script_message(script: SceneScript, speaker: str, current_round: int, total_rounds: int) -> Optional[str]:
    if current_round == 0:
        if speaker == "A" and script.opening_a:
            return script.opening_a
        if speaker == "B" and script.opening_b:
            return script.opening_b
    
    if current_round >= total_rounds - 1:
        if speaker == "A" and script.closing_a:
            return script.closing_a
        if speaker == "B" and script.closing_b:
            return script.closing_b
    
    return None


@app.post("/api/analyze", response_model=CharacterProfile)
async def analyze_character(input_data: TextInput):
    try:
        text = input_data.text
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="文本内容过短，请提供至少10个字符的文本")
        
        keywords = extract_keywords(text, top_n=20)
        mood_particles = extract_mood_particles(text)
        traits = extract_traits(text)
        emotion_tones = analyze_emotion_tones(text)
        speaking_style = determine_speaking_style(mood_particles, traits, emotion_tones)
        system_prompt = generate_system_prompt(
            input_data.name,
            keywords,
            traits,
            emotion_tones,
            speaking_style,
            mood_particles
        )
        
        return CharacterProfile(
            name=input_data.name,
            keywords=keywords,
            mood_particles=mood_particles,
            traits=traits,
            emotion_tones=emotion_tones,
            speaking_style=speaking_style,
            system_prompt=system_prompt
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析过程中发生错误: {str(e)}")


def generate_response(speaker_profile: CharacterProfile, listener_profile: CharacterProfile, 
                      last_message: Optional[str] = None, 
                      context_messages: Optional[List[DialogueMessage]] = None) -> str:
    import random
    
    responses = []
    
    if context_messages and len(context_messages) > 0:
        last_ctx_msg = context_messages[-1]
        responses.append(f"关于你刚才说的'{last_ctx_msg.message[:20]}...'，我有一些想法。")
        responses.append("你说得很有道理，让我想想...")
    
    if speaker_profile.keywords:
        keyword = random.choice(speaker_profile.keywords[:5])
        responses.append(f"关于{keyword}，我有一些想法想要分享。")
        responses.append(f"{keyword}这个话题很有趣，你怎么看？")
    
    if speaker_profile.traits:
        trait = random.choice(speaker_profile.traits[:3])
        if "勇敢" in trait or "自信" in trait:
            responses.append("我觉得我们应该大胆地表达自己的想法。")
            responses.append("让我们坦诚地交流吧，不必有所顾忌。")
        elif "温柔" in trait or "善良" in trait:
            responses.append("我会认真倾听你的每一句话。")
            responses.append("你的想法很重要，我很想听听。")
        elif "聪明" in trait or "智慧" in trait:
            responses.append("让我们从不同角度来思考这个问题。")
            responses.append("这个问题值得我们深入探讨。")
        elif "内向" in trait or "内敛" in trait:
            responses.append("我不太善于表达，但我会努力分享我的想法。")
            responses.append("请给我一点时间整理思路。")
    
    if last_message:
        responses.append(f"你刚才说的话让我想到了很多...")
        responses.append(f"关于你说的内容，我有一些不同的看法。")
        responses.append(f"你的话让我深受启发，让我想想该怎么回应。")
    
    mood_particles = speaker_profile.mood_particles
    if mood_particles:
        particle = random.choice(mood_particles[:3])
        for i in range(len(responses)):
            if not responses[i].endswith(particle):
                responses[i] = responses[i].rstrip("。！？") + particle
    
    if not responses:
        default_responses = [
            "我觉得我们可以聊点有趣的话题。",
            "你最近有什么想法想要分享吗？",
            "让我们开始一段有意义的对话吧。",
            "我很期待与你的交流。",
            "你想讨论什么话题呢？"
        ]
        responses.extend(default_responses)
    
    emotion_tones = speaker_profile.emotion_tones
    dominant_emotion = max(emotion_tones.items(), key=lambda x: x[1])[0]
    
    if dominant_emotion == "positive":
        positive_extras = [
            "这让我感到很开心！",
            "我对这次对话充满期待。",
            "相信我们会有很愉快的交流。"
        ]
        responses.extend(positive_extras)
    elif dominant_emotion == "negative":
        negative_extras = [
            "这个话题让我有些感慨。",
            "我需要认真思考一下。",
            "让我慢慢整理我的思绪。"
        ]
        responses.extend(negative_extras)
    
    return random.choice(responses)


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ShadowPlay API 服务运行中", "version": "2.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "ShadowPlay 服务正常运行"}


async def call_deepseek_api(
    system_prompt: str,
    user_message: str,
    llm_config: LLMConfig
) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Deepseek API 密钥未配置，请设置 DEEPSEEK_API_KEY 环境变量"
        )
    
    url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
        "top_p": llm_config.top_p,
        "frequency_penalty": llm_config.frequency_penalty,
        "presence_penalty": llm_config.presence_penalty
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Deepseek API 调用失败: {response.status} - {error_text}"
                    )
                
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"网络请求失败: {str(e)}"
        )


@app.post("/api/dialogue/llm", response_model=DialogueResponse)
async def generate_llm_dialogue(request: DialogueRequest):
    try:
        speaker_profile = request.character_a_profile if request.speaker == "A" else request.character_b_profile
        listener_profile = request.character_b_profile if request.speaker == "A" else request.character_a_profile
        
        if not request.llm_config:
            request.llm_config = LLMConfig()
        
        system_prompt = speaker_profile.system_prompt
        
        user_message_parts = []
        
        if request.context_messages and len(request.context_messages) > 0:
            context_prompt = build_context_prompt(request.context_messages)
            user_message_parts.append(context_prompt)
        
        if request.last_message:
            user_message_parts.append(f"对方刚刚说：{request.last_message}")
            user_message_parts.append(f"请你以{speaker_profile.name or '说话者'}的身份回应对方。保持你的人格特质，回答要简短有力，富有个性。")
        else:
            user_message_parts.append(f"你是{speaker_profile.name or '说话者'}，正在与{listener_profile.name or '对方'}进行对话。请你先开口，发起一个有趣的话题。保持你的人格特质，回答要简短有力，富有个性。")
        
        user_message = "\n\n".join(user_message_parts)
        
        response = await call_deepseek_api(
            system_prompt,
            user_message,
            request.llm_config
        )
        
        return DialogueResponse(
            message=response,
            speaker=request.speaker,
            generation_mode="llm"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 对话生成过程中发生错误: {str(e)}")


@app.post("/api/dialogue", response_model=DialogueResponse)
async def generate_dialogue(request: DialogueRequest):
    try:
        is_opening = False
        is_closing = False
        script_message = None
        
        if request.use_script and request.scene_script:
            script_message = get_script_message(
                request.scene_script,
                request.speaker,
                request.current_round,
                request.total_rounds
            )
            if script_message:
                if request.current_round == 0:
                    is_opening = True
                elif request.current_round >= request.total_rounds - 1:
                    is_closing = True
        
        if script_message:
            return DialogueResponse(
                message=script_message,
                speaker=request.speaker,
                generation_mode="script",
                is_opening=is_opening,
                is_closing=is_closing
            )
        
        if request.generation_mode == "llm":
            return await generate_llm_dialogue(request)
        
        speaker_profile = request.character_a_profile if request.speaker == "A" else request.character_b_profile
        listener_profile = request.character_b_profile if request.speaker == "A" else request.character_a_profile
        
        response = generate_response(
            speaker_profile,
            listener_profile,
            request.last_message,
            request.context_messages
        )
        
        return DialogueResponse(
            message=response,
            speaker=request.speaker,
            generation_mode="rule"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话生成过程中发生错误: {str(e)}")


PRESET_SCENES = [
    {
        "id": "debate_philosophy",
        "name": "哲学辩论",
        "description": "两位哲学家关于人生意义的激烈辩论",
        "category": "辩论",
        "character_a": {
            "name": "乐观主义者",
            "keywords": ["希望", "未来", "乐观", "积极", "生命", "意义", "幸福", "美好"],
            "mood_particles": ["啊", "呀", "呢"],
            "traits": ["乐观", "热情", "开朗", "勇敢", "自信"],
            "emotion_tones": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            "speaking_style": "语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上",
            "system_prompt": "你是一个坚定的乐观主义者。你相信生命的意义在于追求幸福和美好。你对未来充满希望，总是看到事物光明的一面。你善于用积极的态度感染他人，鼓励人们勇敢面对生活。你的说话风格是：语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上。请保持这个人格特质，与对方进行辩论。"
        },
        "character_b": {
            "name": "悲观主义者",
            "keywords": ["痛苦", "虚无", "悲观", "消极", "死亡", "无意义", "悲伤", "绝望"],
            "mood_particles": ["呢", "吧", "唉"],
            "traits": ["悲观", "冷漠", "内向", "聪明", "成熟"],
            "emotion_tones": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
            "speaking_style": "语气沉稳，善于倾听和回应，性格内敛，深思熟虑，思维敏捷，富有智慧，整体情感较为深沉",
            "system_prompt": "你是一个深刻的悲观主义者。你看透了生命的虚无和痛苦，认为人生本质上是无意义的。你善于深入思考存在的本质，对人类的困境有清醒的认识。你的说话风格是：语气沉稳，善于倾听和回应，性格内敛，深思熟虑，思维敏捷，富有智慧，整体情感较为深沉。请保持这个人格特质，与对方进行辩论。"
        },
        "system_prompt_override": "这是一场关于'人生是否有意义'的哲学辩论。请双方保持各自的人格特质，进行激烈但有深度的思想交锋。",
        "script": {
            "opening_a": "你好！我一直觉得人生充满了无限的可能和美好。你怎么看呢？",
            "opening_b": "你好。坦白说，我对人生的看法可能和你不太一样。让我先听听你的想法吧。",
            "transition_phrases": [
                "关于这个问题，我想从另一个角度来谈谈...",
                "你说得有道理，但我认为...",
                "让我想想怎么回应这个观点...",
                "这让我想到了一个更深层次的问题..."
            ],
            "closing_a": "无论如何，我相信只要我们保持希望，人生就会有意义。谢谢你今天的讨论！",
            "closing_b": "感谢你的分享。虽然我们观点不同，但这场讨论让我思考了很多。也许这就是对话的意义所在。",
            "topic_suggestions": ["人生的意义", "幸福的本质", "痛苦的价值", "希望与绝望"]
        }
    },
    {
        "id": "job_interview",
        "name": "求职面试",
        "description": "一位求职者与面试官之间的专业对话场景",
        "category": "面试",
        "character_a": {
            "name": "面试官",
            "keywords": ["能力", "经验", "团队", "项目", "技能", "责任", "目标", "挑战"],
            "mood_particles": ["吗", "呢", "吧"],
            "traits": ["专业", "严谨", "客观", "聪明", "果断"],
            "emotion_tones": {"positive": 0.3, "negative": 0.1, "neutral": 0.6},
            "speaking_style": "语气沉稳，善于询问和建议，思维敏捷，富有智慧，整体情感中性平衡",
            "system_prompt": "你是一位经验丰富的面试官。你专业、严谨、客观，善于发现候选人的优点和不足。你会提出有深度的问题，考察候选人的真实能力和潜力。你的说话风格是：语气沉稳，善于询问和建议，思维敏捷，富有智慧，整体情感中性平衡。请保持这个人设，进行专业的面试对话。"
        },
        "character_b": {
            "name": "求职者",
            "keywords": ["学习", "成长", "贡献", "热情", "团队", "挑战", "机会", "发展"],
            "mood_particles": ["啊", "呢", "嗯"],
            "traits": ["积极", "热情", "自信", "勤奋", "诚实"],
            "emotion_tones": {"positive": 0.6, "negative": 0.1, "neutral": 0.3},
            "speaking_style": "语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上",
            "system_prompt": "你是一位积极向上的求职者。你对工作充满热情，渴望学习和成长。你诚实、自信，善于展示自己的优点，但也能够正视自己的不足。你希望能够为团队做出贡献，同时实现个人发展。你的说话风格是：语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上。请保持这个人设，进行真诚的面试对话。"
        },
        "system_prompt_override": "这是一场专业的求职面试。面试官请提出考察性问题，求职者请真诚回答。保持专业、礼貌、有深度的对话。",
        "script": {
            "opening_a": "你好！欢迎参加今天的面试。请先简单介绍一下你自己吧。",
            "opening_b": "您好！非常感谢给我这次面试机会。我是一名对工作充满热情的求职者，很高兴能与您交流。",
            "transition_phrases": [
                "很好，那我们来聊聊...",
                "关于这个问题，你能否具体说明一下？",
                "我注意到你提到了...",
                "让我问你一个更具体的问题..."
            ],
            "closing_a": "非常感谢你的分享。我们会尽快通知你面试结果。祝你好运！",
            "closing_b": "非常感谢您的时间和宝贵的问题。我非常期待能有机会加入团队，为公司贡献我的力量。",
            "topic_suggestions": ["自我介绍", "工作经验", "技能特长", "职业规划", "团队协作"]
        }
    },
    {
        "id": "romantic_date",
        "name": "浪漫约会",
        "description": "一对情侣在温馨氛围中的甜蜜对话",
        "category": "社交",
        "character_a": {
            "name": "温柔男生",
            "keywords": ["喜欢", "爱", "开心", "幸福", "期待", "美好", "温暖", "未来"],
            "mood_particles": ["呢", "吧", "呀"],
            "traits": ["温柔", "善良", "体贴", "热情", "自信"],
            "emotion_tones": {"positive": 0.8, "negative": 0.05, "neutral": 0.15},
            "speaking_style": "语气柔和，善于询问和建议，温柔善良，善解人意，整体情感积极向上",
            "system_prompt": "你是一个温柔体贴的男生。你善良、热情，对感情非常认真。你善于观察对方的情绪，懂得如何关心和照顾他人。你说话温柔，懂得浪漫，总是让对方感到温暖和被爱。你的说话风格是：语气柔和，善于询问和建议，温柔善良，善解人意，整体情感积极向上。请保持这个人设，进行甜蜜的约会对话。"
        },
        "character_b": {
            "name": "可爱女生",
            "keywords": ["喜欢", "爱", "开心", "幸福", "期待", "美好", "温暖", "惊喜"],
            "mood_particles": ["啊", "呀", "呢", "啦"],
            "traits": ["可爱", "活泼", "热情", "善良", "温柔"],
            "emotion_tones": {"positive": 0.85, "negative": 0.05, "neutral": 0.1},
            "speaking_style": "语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上",
            "system_prompt": "你是一个可爱活泼的女生。你热情、善良，对生活充满热爱。你善于表达自己的情感，喜欢分享生活中的小确幸。你说话可爱，带有一些撒娇的语气，总是让对方感到愉悦和放松。你的说话风格是：语气活泼，善于表达情感，性格开朗，积极向上，整体情感积极向上。请保持这个人设，进行甜蜜的约会对话。"
        },
        "system_prompt_override": "这是一个浪漫的约会场景。请双方保持甜蜜、温馨的氛围，进行浪漫的对话。可以分享生活趣事、表达情感、畅想未来。",
        "script": {
            "opening_a": "你今天真好看！能和你一起度过这个美好的夜晚，我真的很开心。",
            "opening_b": "你也是呢！我今天特别期待这次约会。你最近过得怎么样呀？",
            "transition_phrases": [
                "对了，我最近发现了一个特别有意思的地方...",
                "你知道吗？每当看到你，我就觉得...",
                "说起这个，让我想起了我们第一次见面的时候...",
                "你有没有想过，我们的未来会是什么样子？"
            ],
            "closing_a": "今晚真的太美好了。我期待着和你一起创造更多这样的时刻。晚安，亲爱的。",
            "closing_b": "我也是！今晚真的很开心。谢谢你给我这么美好的回忆。晚安，梦里见！",
            "topic_suggestions": ["日常趣事", "兴趣爱好", "未来规划", "浪漫回忆", "甜蜜告白"]
        }
    },
    {
        "id": "scientific_debate",
        "name": "科学论战",
        "description": "两位科学家关于前沿科技的激烈讨论",
        "category": "辩论",
        "character_a": {
            "name": "AI 乐观派",
            "keywords": ["人工智能", "创新", "进步", "效率", "自动化", "解放", "可能性", "未来"],
            "mood_particles": ["啊", "呢", "吧"],
            "traits": ["乐观", "创新", "热情", "聪明", "自信"],
            "emotion_tones": {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
            "speaking_style": "语气活泼，善于表达情感，思维敏捷，富有智慧，整体情感积极向上",
            "system_prompt": "你是一个坚定的 AI 乐观派科学家。你相信人工智能将彻底改变人类社会，带来前所未有的进步。你认为 AI 能够解放人类的创造力，解决许多复杂问题。你对技术创新充满热情，总是看到技术的巨大潜力。你的说话风格是：语气活泼，善于表达情感，思维敏捷，富有智慧，整体情感积极向上。请保持这个人设，进行激烈的科学论战。"
        },
        "character_b": {
            "name": "AI 谨慎派",
            "keywords": ["风险", "伦理", "安全", "控制", "责任", "失业", "隐私", "监管"],
            "mood_particles": ["呢", "吧", "嗯"],
            "traits": ["谨慎", "理性", "深思熟虑", "聪明", "成熟"],
            "emotion_tones": {"positive": 0.2, "negative": 0.3, "neutral": 0.5},
            "speaking_style": "语气沉稳，善于倾听和回应，性格内敛，深思熟虑，思维敏捷，富有智慧，整体情感中性平衡",
            "system_prompt": "你是一个谨慎的 AI 伦理学家。你承认人工智能的巨大潜力，但更关注其潜在风险。你认为技术发展必须伴随伦理思考和监管措施。你担心失业、隐私泄露、失控等问题，主张负责任的技术创新。你的说话风格是：语气沉稳，善于倾听和回应，性格内敛，深思熟虑，思维敏捷，富有智慧，整体情感中性平衡。请保持这个人设，进行激烈的科学论战。"
        },
        "system_prompt_override": "这是一场关于'人工智能是否会带来人类的美好未来'的科学论战。请双方保持理性、有深度的讨论，提出具体的论点和论据。",
        "script": {
            "opening_a": "人工智能正在以前所未有的速度发展，我相信它将为人类带来一个更加美好的未来！",
            "opening_b": "我承认 AI 有巨大的潜力，但我们也不能忽视其潜在的风险。让我们理性地讨论这个问题吧。",
            "transition_phrases": [
                "关于这个观点，我想从技术发展的历史来看...",
                "你提到的这个问题确实存在，但我认为...",
                "让我举一个具体的例子来说明...",
                "从伦理层面来看，这个问题需要更深入的思考..."
            ],
            "closing_a": "无论如何，我相信只要我们保持乐观和创新精神，AI 一定会为人类带来福祉。",
            "closing_b": "我同意创新的重要性，但我仍然认为我们需要谨慎前行。感谢这场有意义的讨论。",
            "topic_suggestions": ["AI 伦理", "技术风险", "就业影响", "监管政策", "未来愿景"]
        }
    }
]


def init_preset_scenes():
    for scene in PRESET_SCENES:
        scene_path = os.path.join(DATA_DIR, "scenes", f"{scene['id']}.json")
        if not os.path.exists(scene_path):
            scene_data = {
                "id": scene["id"],
                "name": scene["name"],
                "description": scene["description"],
                "category": scene["category"],
                "character_a_profile": scene["character_a"],
                "character_b_profile": scene["character_b"],
                "system_prompt_override": scene.get("system_prompt_override"),
                "script": scene.get("script"),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene_data, f, ensure_ascii=False, indent=2)


@app.get("/api/scenes")
async def list_scenes():
    try:
        init_preset_scenes()
        scenes = []
        scenes_dir = os.path.join(DATA_DIR, "scenes")
        for filename in os.listdir(scenes_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(scenes_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    scene = json.load(f)
                scenes.append({
                    "id": scene.get("id"),
                    "name": scene.get("name"),
                    "description": scene.get("description"),
                    "category": scene.get("category"),
                    "character_a_name": scene.get("character_a_profile", {}).get("name"),
                    "character_b_name": scene.get("character_b_profile", {}).get("name"),
                    "has_script": scene.get("script") is not None,
                    "created_at": scene.get("created_at"),
                    "updated_at": scene.get("updated_at")
                })
        return {"scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取场景列表失败: {str(e)}")


@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str):
    try:
        scene_path = os.path.join(DATA_DIR, "scenes", f"{scene_id}.json")
        if not os.path.exists(scene_path):
            raise HTTPException(status_code=404, detail="场景不存在")
        
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
        return scene
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取场景失败: {str(e)}")


@app.post("/api/scenes")
async def create_scene(scene_data: SceneTemplateCreate):
    try:
        scene_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        new_scene = {
            "id": scene_id,
            "name": scene_data.name,
            "description": scene_data.description,
            "category": scene_data.category,
            "character_a_profile": scene_data.character_a_profile.dict(),
            "character_b_profile": scene_data.character_b_profile.dict(),
            "system_prompt_override": scene_data.system_prompt_override,
            "script": scene_data.script.dict() if scene_data.script else None,
            "created_at": now,
            "updated_at": now
        }
        
        file_path = os.path.join(DATA_DIR, "scenes", f"{scene_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(new_scene, f, ensure_ascii=False, indent=2)
        
        return new_scene
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建场景失败: {str(e)}")


@app.put("/api/scenes/{scene_id}")
async def update_scene(scene_id: str, scene_data: SceneTemplateUpdate):
    try:
        file_path = os.path.join(DATA_DIR, "scenes", f"{scene_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="场景不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            existing_scene = json.load(f)
        
        if scene_data.name is not None:
            existing_scene["name"] = scene_data.name
        if scene_data.description is not None:
            existing_scene["description"] = scene_data.description
        if scene_data.category is not None:
            existing_scene["category"] = scene_data.category
        if scene_data.character_a_profile is not None:
            existing_scene["character_a_profile"] = scene_data.character_a_profile.dict()
        if scene_data.character_b_profile is not None:
            existing_scene["character_b_profile"] = scene_data.character_b_profile.dict()
        if scene_data.system_prompt_override is not None:
            existing_scene["system_prompt_override"] = scene_data.system_prompt_override
        if scene_data.script is not None:
            existing_scene["script"] = scene_data.script.dict()
        
        existing_scene["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_scene, f, ensure_ascii=False, indent=2)
        
        return existing_scene
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新场景失败: {str(e)}")


@app.delete("/api/scenes/{scene_id}")
async def delete_scene(scene_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "scenes", f"{scene_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="场景不存在")
        
        os.remove(file_path)
        return {"message": "场景已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除场景失败: {str(e)}")


@app.get("/api/characters")
async def list_characters():
    try:
        characters = []
        chars_dir = os.path.join(DATA_DIR, "characters")
        for filename in os.listdir(chars_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(chars_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    char = json.load(f)
                characters.append({
                    "id": char["id"],
                    "name": char["name"],
                    "created_at": char["created_at"],
                    "updated_at": char["updated_at"]
                })
        return {"characters": characters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取人物列表失败: {str(e)}")


@app.post("/api/characters")
async def create_character(name: str, profile: CharacterProfile):
    try:
        char_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        char_data = {
            "id": char_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "profile": profile.dict()
        }
        
        file_path = os.path.join(DATA_DIR, "characters", f"{char_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, ensure_ascii=False, indent=2)
        
        return char_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建人物失败: {str(e)}")


@app.get("/api/characters/{char_id}")
async def get_character(char_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "characters", f"{char_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="人物不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            char = json.load(f)
        return char
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取人物失败: {str(e)}")


@app.put("/api/characters/{char_id}")
async def update_character(char_id: str, name: str, profile: CharacterProfile):
    try:
        file_path = os.path.join(DATA_DIR, "characters", f"{char_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="人物不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            char = json.load(f)
        
        char["name"] = name
        char["updated_at"] = datetime.now().isoformat()
        char["profile"] = profile.dict()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char, f, ensure_ascii=False, indent=2)
        
        return char
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新人物失败: {str(e)}")


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "characters", f"{char_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="人物不存在")
        
        os.remove(file_path)
        return {"message": "人物已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除人物失败: {str(e)}")


@app.post("/api/characters/import")
async def import_characters(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="导入的数据格式错误，应为数组")
        
        imported = []
        for item in data:
            if "name" not in item or "profile" not in item:
                continue
            
            char_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            char_data = {
                "id": char_id,
                "name": item["name"],
                "created_at": now,
                "updated_at": now,
                "profile": item["profile"]
            }
            
            file_path = os.path.join(DATA_DIR, "characters", f"{char_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(char_data, f, ensure_ascii=False, indent=2)
            
            imported.append(char_data)
        
        return {"imported": imported, "count": len(imported)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON 格式错误")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入人物失败: {str(e)}")


@app.get("/api/characters/export")
async def export_characters():
    try:
        characters = []
        chars_dir = os.path.join(DATA_DIR, "characters")
        for filename in os.listdir(chars_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(chars_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    char = json.load(f)
                characters.append({
                    "name": char["name"],
                    "profile": char["profile"]
                })
        
        return JSONResponse(
            content=characters,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=characters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出人物失败: {str(e)}")


@app.post("/api/dialogues")
async def create_dialogue(
    character_a_name: str = Query(...),
    character_b_name: str = Query(...),
    scene_id: Optional[str] = Query(None),
    scene_name: Optional[str] = Query(None),
    title: Optional[str] = Query(None)
):
    try:
        dialogue_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        dialogue_title = title or f"{character_a_name} vs {character_b_name}"
        
        dialogue_data = {
            "id": dialogue_id,
            "title": dialogue_title,
            "character_a_name": character_a_name,
            "character_b_name": character_b_name,
            "scene_id": scene_id,
            "scene_name": scene_name,
            "messages": [],
            "context_summary": None,
            "created_at": now,
            "updated_at": now,
            "total_rounds": 0
        }
        
        file_path = os.path.join(DATA_DIR, "dialogues", f"{dialogue_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
        
        return dialogue_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建对话记录失败: {str(e)}")


@app.get("/api/dialogues")
async def list_dialogues():
    try:
        dialogues = []
        dialogues_dir = os.path.join(DATA_DIR, "dialogues")
        for filename in os.listdir(dialogues_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(dialogues_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    dialogue = json.load(f)
                dialogues.append({
                    "id": dialogue["id"],
                    "title": dialogue["title"],
                    "character_a_name": dialogue["character_a_name"],
                    "character_b_name": dialogue["character_b_name"],
                    "scene_name": dialogue.get("scene_name"),
                    "total_rounds": dialogue["total_rounds"],
                    "created_at": dialogue["created_at"],
                    "updated_at": dialogue["updated_at"]
                })
        
        dialogues.sort(key=lambda x: x["created_at"], reverse=True)
        return {"dialogues": dialogues}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {str(e)}")


@app.get("/api/dialogues/{dialogue_id}")
async def get_dialogue(dialogue_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "dialogues", f"{dialogue_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="对话记录不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            dialogue = json.load(f)
        return dialogue
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话记录失败: {str(e)}")


@app.post("/api/dialogues/{dialogue_id}/messages")
async def add_dialogue_message(dialogue_id: str, message: DialogueMessage):
    try:
        file_path = os.path.join(DATA_DIR, "dialogues", f"{dialogue_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="对话记录不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            dialogue = json.load(f)
        
        dialogue["messages"].append(message.dict())
        dialogue["total_rounds"] = max(dialogue["total_rounds"], message.round_number + 1)
        dialogue["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        return dialogue
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加对话消息失败: {str(e)}")


@app.delete("/api/dialogues/{dialogue_id}")
async def delete_dialogue(dialogue_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "dialogues", f"{dialogue_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="对话记录不存在")
        
        os.remove(file_path)
        return {"message": "对话记录已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除对话记录失败: {str(e)}")


@app.get("/api/dialogues/{dialogue_id}/export")
async def export_dialogue(dialogue_id: str):
    try:
        file_path = os.path.join(DATA_DIR, "dialogues", f"{dialogue_id}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="对话记录不存在")
        
        with open(file_path, "r", encoding="utf-8") as f:
            dialogue = json.load(f)
        
        return JSONResponse(
            content=dialogue,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=dialogue_{dialogue_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出对话记录失败: {str(e)}")


@app.get("/api/llm/status")
async def check_llm_status():
    return {
        "configured": bool(DEEPSEEK_API_KEY),
        "model": DEEPSEEK_MODEL,
        "api_base": DEEPSEEK_API_BASE
    }


if __name__ == "__main__":
    init_preset_scenes()
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
