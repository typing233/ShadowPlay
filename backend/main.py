from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import os
from collections import Counter

app = FastAPI(title="ShadowPlay - 人格复刻与思想对撞沙盒", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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


class DialogueRequest(BaseModel):
    character_a_profile: CharacterProfile
    character_b_profile: CharacterProfile
    last_message: Optional[str] = None
    speaker: str


class DialogueResponse(BaseModel):
    message: str
    speaker: str


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


@app.post("/api/dialogue", response_model=DialogueResponse)
async def generate_dialogue(request: DialogueRequest):
    try:
        speaker_profile = request.character_a_profile if request.speaker == "A" else request.character_b_profile
        listener_profile = request.character_b_profile if request.speaker == "A" else request.character_a_profile
        
        response = generate_response(
            speaker_profile,
            listener_profile,
            request.last_message
        )
        
        return DialogueResponse(
            message=response,
            speaker=request.speaker
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话生成过程中发生错误: {str(e)}")


def generate_response(speaker_profile: CharacterProfile, listener_profile: CharacterProfile, 
                      last_message: Optional[str] = None) -> str:
    import random
    
    responses = []
    
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
    return {"message": "ShadowPlay API 服务运行中", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "ShadowPlay 服务正常运行"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
