"""
Response Analyzer Module
지원자 응답을 분석하고 피드백을 생성하는 모듈
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import asyncio
from datetime import datetime

from loguru import logger
from openai import AsyncOpenAI

from config.settings import get_settings


class ResponseQuality(Enum):
    """응답 품질 등급"""
    EXCELLENT = "excellent"  # 90-100점
    GOOD = "good"  # 70-89점
    FAIR = "fair"  # 50-69점
    POOR = "poor"  # 0-49점


@dataclass
class ResponseAnalysis:
    """응답 분석 결과"""
    # 기본 정보
    question_id: str
    question_text: str
    answer_text: str
    timestamp: datetime

    # 분석 결과
    word_count: int
    char_count: int
    sentence_count: int

    # STAR 구조 분석
    has_situation: bool = False
    has_task: bool = False
    has_action: bool = False
    has_result: bool = False
    star_score: float = 0.0  # 0.0 ~ 1.0

    # 구체성 분석
    has_specific_examples: bool = False
    has_numbers_metrics: bool = False
    has_concrete_details: bool = False
    specificity_score: float = 0.0  # 0.0 ~ 1.0

    # 키워드 분석
    technical_keywords: List[str] = field(default_factory=list)
    soft_skill_keywords: List[str] = field(default_factory=list)
    action_verbs: List[str] = field(default_factory=list)

    # 감정/톤 분석
    confidence_level: str = "medium"  # low, medium, high
    enthusiasm_level: str = "medium"  # low, medium, high
    professionalism_score: float = 0.0  # 0.0 ~ 1.0

    # 종합 평가
    overall_score: float = 0.0  # 0.0 ~ 100.0
    quality: ResponseQuality = ResponseQuality.FAIR

    # 실시간 피드백
    needs_followup: bool = False
    followup_suggestions: List[str] = field(default_factory=list)

    # 강점 및 개선점
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


@dataclass
class InterviewReport:
    """면접 종합 리포트"""
    candidate_name: Optional[str]
    interview_date: datetime
    total_questions: int
    total_duration_minutes: float

    # 응답별 분석
    responses: List[ResponseAnalysis]

    # 종합 평가
    average_score: float
    overall_quality: ResponseQuality

    # 카테고리별 점수
    category_scores: Dict[str, float]  # category_name -> score

    # 강점 및 개선점 (종합)
    key_strengths: List[str]
    key_improvements: List[str]

    # 추천 사항
    hiring_recommendation: str  # strong_yes, yes, maybe, no, strong_no
    recommendation_reason: str

    # 상세 분석
    star_analysis: str
    communication_analysis: str
    technical_analysis: str


class ResponseAnalyzer:
    """지원자 응답 분석기"""

    # STAR 키워드
    SITUATION_KEYWORDS = [
        "상황", "당시", "프로젝트", "회사", "팀", "고객", "문제가",
        "어려움", "도전", "환경", "배경"
    ]

    TASK_KEYWORDS = [
        "역할", "담당", "책임", "맡았", "목표", "과제", "업무",
        "해야", "필요", "요구"
    ]

    ACTION_KEYWORDS = [
        "했습니다", "진행했", "수행했", "개발했", "구현했", "설계했",
        "분석했", "해결했", "개선했", "작성했", "적용했", "도입했",
        "시도했", "노력했", "협업했"
    ]

    RESULT_KEYWORDS = [
        "결과", "성과", "달성", "개선", "증가", "감소", "향상",
        "성공", "완료", "%", "배", "시간", "비용"
    ]

    # 기술 키워드 (예시)
    TECHNICAL_KEYWORDS = [
        "python", "java", "javascript", "react", "django", "flask",
        "api", "database", "sql", "nosql", "docker", "kubernetes",
        "aws", "gcp", "azure", "git", "ci/cd", "microservice",
        "restful", "graphql", "redis", "mongodb", "postgresql",
        "machine learning", "ai", "deep learning", "tensorflow"
    ]

    # 소프트 스킬 키워드
    SOFT_SKILL_KEYWORDS = [
        "협업", "커뮤니케이션", "리더십", "문제해결", "창의성",
        "책임감", "적극성", "열정", "배움", "성장", "도전",
        "팀워크", "조율", "설득", "멘토링"
    ]

    # 행동 동사
    ACTION_VERBS = [
        "개발", "구현", "설계", "분석", "해결", "개선", "최적화",
        "작성", "관리", "운영", "배포", "테스트", "검증", "협의",
        "제안", "도입", "적용", "연구", "학습"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 가져옴)
        """
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.llm_model

        logger.info("ResponseAnalyzer initialized")

    async def analyze_response(
        self,
        question_id: str,
        question_text: str,
        answer_text: str,
        use_llm: bool = True
    ) -> ResponseAnalysis:
        """
        응답 분석

        Args:
            question_id: 질문 ID
            question_text: 질문 텍스트
            answer_text: 응답 텍스트
            use_llm: LLM 사용 여부 (False면 규칙 기반만)

        Returns:
            ResponseAnalysis: 분석 결과
        """
        # 기본 분석 (규칙 기반)
        analysis = ResponseAnalysis(
            question_id=question_id,
            question_text=question_text,
            answer_text=answer_text,
            timestamp=datetime.now(),
            word_count=len(answer_text.split()),
            char_count=len(answer_text),
            sentence_count=len([s for s in answer_text.split('.') if s.strip()])
        )

        # STAR 구조 분석
        self._analyze_star_structure(answer_text, analysis)

        # 구체성 분석
        self._analyze_specificity(answer_text, analysis)

        # 키워드 분석
        self._analyze_keywords(answer_text, analysis)

        # LLM 기반 심층 분석
        if use_llm:
            await self._llm_deep_analysis(question_text, answer_text, analysis)

        # 종합 점수 계산
        self._calculate_overall_score(analysis)

        # 실시간 피드백 생성
        self._generate_realtime_feedback(analysis)

        logger.info(
            f"Response analyzed: Q={question_id}, "
            f"Score={analysis.overall_score:.1f}, "
            f"Quality={analysis.quality.value}"
        )

        return analysis

    def _analyze_star_structure(self, text: str, analysis: ResponseAnalysis):
        """STAR 구조 분석"""
        text_lower = text.lower()

        # 각 요소 감지
        analysis.has_situation = any(kw in text for kw in self.SITUATION_KEYWORDS)
        analysis.has_task = any(kw in text for kw in self.TASK_KEYWORDS)
        analysis.has_action = any(kw in text for kw in self.ACTION_KEYWORDS)
        analysis.has_result = any(kw in text for kw in self.RESULT_KEYWORDS)

        # STAR 점수 계산 (0.0 ~ 1.0)
        star_components = [
            analysis.has_situation,
            analysis.has_task,
            analysis.has_action,
            analysis.has_result
        ]
        analysis.star_score = sum(star_components) / 4.0

    def _analyze_specificity(self, text: str, analysis: ResponseAnalysis):
        """구체성 분석"""
        # 숫자/메트릭 포함 여부
        analysis.has_numbers_metrics = bool(re.search(r'\d+', text))

        # 구체적인 예시 (길이 기반)
        analysis.has_specific_examples = len(text) > 100

        # 구체적인 세부사항 (문장 수 기반)
        analysis.has_concrete_details = analysis.sentence_count >= 3

        # 구체성 점수
        specificity_components = [
            analysis.has_numbers_metrics,
            analysis.has_specific_examples,
            analysis.has_concrete_details
        ]
        analysis.specificity_score = sum(specificity_components) / 3.0

    def _analyze_keywords(self, text: str, analysis: ResponseAnalysis):
        """키워드 분석"""
        text_lower = text.lower()

        # 기술 키워드
        analysis.technical_keywords = [
            kw for kw in self.TECHNICAL_KEYWORDS
            if kw.lower() in text_lower
        ]

        # 소프트 스킬 키워드
        analysis.soft_skill_keywords = [
            kw for kw in self.SOFT_SKILL_KEYWORDS
            if kw in text
        ]

        # 행동 동사
        analysis.action_verbs = [
            verb for verb in self.ACTION_VERBS
            if verb in text
        ]

    async def _llm_deep_analysis(
        self,
        question: str,
        answer: str,
        analysis: ResponseAnalysis
    ):
        """LLM 기반 심층 분석"""
        prompt = f"""다음 면접 질문과 답변을 분석해주세요.

질문: {question}

답변: {answer}

다음 항목을 JSON 형식으로 평가해주세요:
1. confidence_level: 자신감 수준 (low/medium/high)
2. enthusiasm_level: 열정 수준 (low/medium/high)
3. professionalism_score: 전문성 점수 (0.0 ~ 1.0)
4. strengths: 강점 리스트 (최대 3개)
5. improvements: 개선점 리스트 (최대 3개)

JSON만 출력하세요."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )

            # JSON 파싱
            import json
            result = json.loads(response.choices[0].message.content)

            analysis.confidence_level = result.get("confidence_level", "medium")
            analysis.enthusiasm_level = result.get("enthusiasm_level", "medium")
            analysis.professionalism_score = result.get("professionalism_score", 0.5)
            analysis.strengths = result.get("strengths", [])
            analysis.improvements = result.get("improvements", [])

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            # 폴백: 기본값 사용
            analysis.confidence_level = "medium"
            analysis.enthusiasm_level = "medium"
            analysis.professionalism_score = 0.5

    def _calculate_overall_score(self, analysis: ResponseAnalysis):
        """종합 점수 계산"""
        # 가중치 기반 점수 계산
        weights = {
            "star": 0.3,  # STAR 구조 30%
            "specificity": 0.25,  # 구체성 25%
            "professionalism": 0.15,  # 전문성 15%
            "keywords": 0.15,  # 키워드 15%
            "length": 0.15  # 길이 적절성 15%
        }

        # 각 항목 점수 (0 ~ 100)
        star_score = analysis.star_score * 100
        specificity_score = analysis.specificity_score * 100
        professionalism_score = analysis.professionalism_score * 100

        # 키워드 점수 (기술 + 소프트스킬)
        keyword_count = len(analysis.technical_keywords) + len(analysis.soft_skill_keywords)
        keyword_score = min(keyword_count * 10, 100)

        # 길이 적절성 점수 (50~300 단어가 이상적)
        if analysis.word_count < 20:
            length_score = 30
        elif analysis.word_count < 50:
            length_score = 60
        elif 50 <= analysis.word_count <= 300:
            length_score = 100
        elif analysis.word_count <= 500:
            length_score = 80
        else:
            length_score = 60

        # 가중 평균
        analysis.overall_score = (
            star_score * weights["star"] +
            specificity_score * weights["specificity"] +
            professionalism_score * weights["professionalism"] +
            keyword_score * weights["keywords"] +
            length_score * weights["length"]
        )

        # 품질 등급 결정
        if analysis.overall_score >= 90:
            analysis.quality = ResponseQuality.EXCELLENT
        elif analysis.overall_score >= 70:
            analysis.quality = ResponseQuality.GOOD
        elif analysis.overall_score >= 50:
            analysis.quality = ResponseQuality.FAIR
        else:
            analysis.quality = ResponseQuality.POOR

    def _generate_realtime_feedback(self, analysis: ResponseAnalysis):
        """실시간 피드백 생성"""
        suggestions = []

        # 너무 짧은 답변
        if analysis.word_count < 20:
            analysis.needs_followup = True
            suggestions.append("조금 더 구체적으로 설명해주실 수 있을까요?")

        # STAR 구조 부족
        if analysis.star_score < 0.5:
            analysis.needs_followup = True
            if not analysis.has_situation:
                suggestions.append("어떤 상황이었는지 배경을 설명해주시겠어요?")
            if not analysis.has_action:
                suggestions.append("구체적으로 어떤 행동을 취하셨나요?")
            if not analysis.has_result:
                suggestions.append("그 결과는 어떻게 되었나요?")

        # 구체성 부족
        if not analysis.has_numbers_metrics and analysis.star_score >= 0.5:
            suggestions.append("혹시 구체적인 수치나 결과 지표를 말씀해주실 수 있나요?")

        # 기술 키워드 부족 (기술 질문인 경우)
        if len(analysis.technical_keywords) == 0 and "기술" in analysis.question_text:
            suggestions.append("사용하신 기술 스택이나 도구에 대해 말씀해주시겠어요?")

        analysis.followup_suggestions = suggestions[:2]  # 최대 2개

    async def generate_interview_report(
        self,
        responses: List[ResponseAnalysis],
        candidate_name: Optional[str] = None,
        category_mapping: Optional[Dict[str, str]] = None
    ) -> InterviewReport:
        """
        면접 종합 리포트 생성

        Args:
            responses: 응답 분석 결과 리스트
            candidate_name: 지원자 이름
            category_mapping: 질문 ID -> 카테고리 매핑

        Returns:
            InterviewReport: 면접 종합 리포트
        """
        if not responses:
            raise ValueError("No responses to analyze")

        # 기본 정보
        interview_date = responses[0].timestamp
        total_questions = len(responses)
        duration = (responses[-1].timestamp - responses[0].timestamp).total_seconds() / 60

        # 평균 점수
        average_score = sum(r.overall_score for r in responses) / len(responses)

        # 전체 품질 등급
        if average_score >= 90:
            overall_quality = ResponseQuality.EXCELLENT
        elif average_score >= 70:
            overall_quality = ResponseQuality.GOOD
        elif average_score >= 50:
            overall_quality = ResponseQuality.FAIR
        else:
            overall_quality = ResponseQuality.POOR

        # 카테고리별 점수
        category_scores = {}
        if category_mapping:
            for category in set(category_mapping.values()):
                category_responses = [
                    r for r in responses
                    if category_mapping.get(r.question_id) == category
                ]
                if category_responses:
                    category_scores[category] = sum(
                        r.overall_score for r in category_responses
                    ) / len(category_responses)

        # 종합 강점 및 개선점
        all_strengths = []
        all_improvements = []
        for r in responses:
            all_strengths.extend(r.strengths)
            all_improvements.extend(r.improvements)

        # 빈도 기반 상위 3개
        from collections import Counter
        key_strengths = [s for s, _ in Counter(all_strengths).most_common(3)]
        key_improvements = [i for i, _ in Counter(all_improvements).most_common(3)]

        # 채용 추천
        if average_score >= 85:
            recommendation = "strong_yes"
            reason = "모든 영역에서 탁월한 역량을 보여주셨습니다."
        elif average_score >= 70:
            recommendation = "yes"
            reason = "전반적으로 우수한 역량을 갖추고 계십니다."
        elif average_score >= 55:
            recommendation = "maybe"
            reason = "일부 영역에서 보완이 필요하지만 잠재력이 있습니다."
        elif average_score >= 40:
            recommendation = "no"
            reason = "현재 요구 역량에 다소 미치지 못합니다."
        else:
            recommendation = "strong_no"
            reason = "현재로서는 채용을 권장하기 어렵습니다."

        # LLM 기반 상세 분석
        star_analysis = await self._generate_star_analysis(responses)
        communication_analysis = await self._generate_communication_analysis(responses)
        technical_analysis = await self._generate_technical_analysis(responses)

        report = InterviewReport(
            candidate_name=candidate_name,
            interview_date=interview_date,
            total_questions=total_questions,
            total_duration_minutes=duration,
            responses=responses,
            average_score=average_score,
            overall_quality=overall_quality,
            category_scores=category_scores,
            key_strengths=key_strengths,
            key_improvements=key_improvements,
            hiring_recommendation=recommendation,
            recommendation_reason=reason,
            star_analysis=star_analysis,
            communication_analysis=communication_analysis,
            technical_analysis=technical_analysis
        )

        logger.info(
            f"Interview report generated: "
            f"Score={average_score:.1f}, "
            f"Recommendation={recommendation}"
        )

        return report

    async def _generate_star_analysis(self, responses: List[ResponseAnalysis]) -> str:
        """STAR 구조 종합 분석"""
        avg_star = sum(r.star_score for r in responses) / len(responses)

        if avg_star >= 0.8:
            return "지원자는 대부분의 답변에서 STAR 구조를 잘 활용하여 경험을 체계적으로 설명했습니다."
        elif avg_star >= 0.5:
            return "지원자는 경험을 설명할 때 상황과 행동을 언급하지만, 결과나 성과 측면을 보완하면 더 좋겠습니다."
        else:
            return "답변이 다소 추상적이며, 구체적인 상황-행동-결과 구조를 갖추면 설득력이 높아질 것입니다."

    async def _generate_communication_analysis(self, responses: List[ResponseAnalysis]) -> str:
        """커뮤니케이션 역량 분석"""
        avg_length = sum(r.word_count for r in responses) / len(responses)
        high_confidence_count = sum(1 for r in responses if r.confidence_level == "high")

        if avg_length > 100 and high_confidence_count >= len(responses) * 0.6:
            return "명확하고 자신감 있는 커뮤니케이션 스타일을 보여줍니다."
        elif avg_length < 30:
            return "답변이 다소 짧은 편이며, 더 풍부한 설명이 필요합니다."
        else:
            return "적절한 수준의 커뮤니케이션 역량을 갖추고 있습니다."

    async def _generate_technical_analysis(self, responses: List[ResponseAnalysis]) -> str:
        """기술 역량 분석"""
        all_keywords = []
        for r in responses:
            all_keywords.extend(r.technical_keywords)

        unique_keywords = set(all_keywords)

        if len(unique_keywords) >= 10:
            return f"다양한 기술 스택({len(unique_keywords)}개)을 경험하고 이해하고 있습니다."
        elif len(unique_keywords) >= 5:
            return "적절한 수준의 기술적 경험을 보유하고 있습니다."
        else:
            return "기술적 경험에 대한 구체적인 언급이 부족합니다."


# 헬퍼 함수
def create_response_analyzer(api_key: Optional[str] = None) -> ResponseAnalyzer:
    """ResponseAnalyzer 인스턴스 생성"""
    return ResponseAnalyzer(api_key=api_key)


# 사용 예시
if __name__ == "__main__":
    async def main():
        # 분석기 생성
        analyzer = create_response_analyzer()

        # 샘플 질문 및 답변
        question = "가장 어려웠던 프로젝트 경험에 대해 말씀해주세요."
        answer = """
        제가 가장 어려웠던 프로젝트는 대용량 트래픽을 처리하는 API 서버를 개발했던 경험입니다.
        당시 기존 시스템이 초당 1000건의 요청만 처리할 수 있었는데,
        마케팅 캠페인으로 인해 예상 트래픽이 10배 증가할 것으로 예측되었습니다.

        저는 팀 리드로서 시스템 아키텍처를 재설계하는 역할을 맡았습니다.
        먼저 병목 지점을 분석했고, Redis 캐싱 레이어를 도입하고
        데이터베이스 쿼리를 최적화했습니다. 또한 수평 확장이 가능하도록
        마이크로서비스 아키텍처로 전환했습니다.

        그 결과, 캠페인 기간 동안 초당 15000건의 요청을 안정적으로 처리했고,
        응답 시간도 평균 500ms에서 50ms로 10배 개선되었습니다.
        """

        # 응답 분석
        analysis = await analyzer.analyze_response(
            question_id="q1",
            question_text=question,
            answer_text=answer
        )

        print(f"=== 응답 분석 결과 ===")
        print(f"종합 점수: {analysis.overall_score:.1f}")
        print(f"품질: {analysis.quality.value}")
        print(f"STAR 점수: {analysis.star_score:.2f}")
        print(f"구체성 점수: {analysis.specificity_score:.2f}")
        print(f"기술 키워드: {analysis.technical_keywords}")
        print(f"강점: {analysis.strengths}")
        print(f"개선점: {analysis.improvements}")

        if analysis.needs_followup:
            print(f"\n추가 질문 제안:")
            for suggestion in analysis.followup_suggestions:
                print(f"  - {suggestion}")

    asyncio.run(main())
