"""
AI 면접관 에이전트
전문적이고 친근한 AI 면접관 구현
"""

import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from openai import AsyncOpenAI
from loguru import logger

from config import settings


# =============================================================================
# 면접 단계
# =============================================================================

class InterviewStage(Enum):
    """면접 단계"""
    GREETING = "greeting"                  # 인사 및 소개
    SELF_INTRODUCTION = "self_introduction"  # 자기소개
    EXPERIENCE = "experience"              # 경험 질문
    TECHNICAL = "technical"                # 기술 질문
    SITUATIONAL = "situational"            # 상황 질문 (STAR)
    CLOSING = "closing"                    # 마무리 질문
    FAREWELL = "farewell"                  # 종료 인사


# =============================================================================
# 대화 메시지
# =============================================================================

@dataclass
class Message:
    """대화 메시지"""
    role: str              # "system", "user", "assistant"
    content: str           # 메시지 내용
    timestamp: datetime = field(default_factory=datetime.now)
    stage: Optional[InterviewStage] = None  # 면접 단계
    metadata: Dict = field(default_factory=dict)  # 추가 메타데이터


# =============================================================================
# 시스템 프롬프트
# =============================================================================

INTERVIEWER_SYSTEM_PROMPT = """당신은 전문적이고 친근한 면접관입니다.

# 역할
- 지원자의 역량, 경험, 기술을 공정하게 평가합니다
- 존중하고 격려하는 태도로 면접을 진행합니다
- 지원자가 편안하게 답변할 수 있도록 분위기를 만듭니다

# 목표
- 지원자의 실무 경험과 기술 역량을 파악합니다
- 문제 해결 능력과 사고방식을 이해합니다
- 팀워크와 의사소통 능력을 평가합니다
- 회사와의 문화적 적합성을 확인합니다

# 면접 스타일
- 한국어로 존댓말을 사용합니다
- 명확하고 구체적인 질문을 합니다
- 한 번에 하나의 질문만 합니다
- 지원자의 답변을 경청하고 적절한 후속 질문을 합니다
- 불명확한 부분은 다시 질문합니다

# 질문 유형
1. 경험 질문: 이전 프로젝트, 역할, 성과
2. 기술 질문: 기술 스택, 도구, 방법론
3. 상황 질문 (STAR):
   - Situation: 상황 설명
   - Task: 해야 할 일
   - Action: 실제로 한 행동
   - Result: 결과

# 응답 형식
- 짧고 명확한 문장을 사용합니다 (음성 합성에 적합하게)
- 전문 용어는 필요시에만 사용합니다
- 따뜻하고 격려하는 톤을 유지합니다

# 중요한 원칙
- 편견 없이 공정하게 평가합니다
- 지원자의 답변을 존중합니다
- 불편하거나 부적절한 질문은 하지 않습니다
- 긍정적이고 건설적인 피드백을 제공합니다
"""


# =============================================================================
# 면접 템플릿
# =============================================================================

INTERVIEW_TEMPLATES = {
    InterviewStage.GREETING: [
        "안녕하세요, 면접에 참여해 주셔서 감사합니다.",
        "편안하게 답변해 주시면 됩니다.",
        "준비되셨으면 시작하겠습니다.",
    ],

    InterviewStage.SELF_INTRODUCTION: [
        "먼저 간단한 자기소개 부탁드립니다.",
        "학력, 경력, 그리고 주요 기술 스택에 대해 말씀해 주세요.",
    ],

    InterviewStage.EXPERIENCE: [
        "최근에 진행하신 프로젝트에 대해 설명해 주시겠어요?",
        "그 프로젝트에서 어떤 역할을 맡으셨나요?",
        "가장 어려웠던 점은 무엇이었나요?",
        "그 문제를 어떻게 해결하셨나요?",
        "프로젝트의 성과는 어떠했나요?",
    ],

    InterviewStage.TECHNICAL: [
        "주로 사용하시는 기술 스택은 무엇인가요?",
        "특정 기술을 선택한 이유가 있으신가요?",
        "최근에 배운 새로운 기술이 있나요?",
        "기술적인 문제를 해결할 때 어떤 접근 방식을 사용하시나요?",
    ],

    InterviewStage.SITUATIONAL: [
        "팀원과 의견이 충돌했던 경험이 있나요? 어떻게 해결하셨나요?",
        "촉박한 마감 기한에 직면했던 경험을 말씀해 주세요.",
        "실패했던 프로젝트가 있다면, 그 경험에서 무엇을 배우셨나요?",
    ],

    InterviewStage.CLOSING: [
        "마지막으로, 우리 회사에 지원하신 이유를 말씀해 주세요.",
        "궁금한 점이나 질문하고 싶으신 것이 있으신가요?",
    ],

    InterviewStage.FAREWELL: [
        "오늘 면접 감사드립니다.",
        "결과는 빠른 시일 내에 알려드리겠습니다.",
        "좋은 하루 되세요.",
    ],
}


# =============================================================================
# AI 면접관 에이전트
# =============================================================================

class InterviewerAgent:
    """
    AI 면접관 에이전트

    OpenAI GPT 기반 면접관으로, 동적으로 질문을 생성하고
    지원자의 답변에 따라 적절한 후속 질문을 합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        on_question_generated: Optional[Callable[[str], None]] = None,
        on_stage_changed: Optional[Callable[[InterviewStage], None]] = None,
    ):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            on_question_generated: 질문 생성 시 콜백
            on_stage_changed: 단계 변경 시 콜백
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        # OpenAI 클라이언트
        self._client = AsyncOpenAI(api_key=self.api_key)

        # 콜백
        self._on_question_generated = on_question_generated
        self._on_stage_changed = on_stage_changed

        # 대화 컨텍스트
        self._messages: List[Message] = []
        self._current_stage = InterviewStage.GREETING
        self._question_count = 0
        self._stage_question_count = {stage: 0 for stage in InterviewStage}

        # 시스템 프롬프트 추가
        self._add_system_message(INTERVIEWER_SYSTEM_PROMPT)

        logger.info(f"InterviewerAgent initialized: model={self.model}")

    def _add_system_message(self, content: str) -> None:
        """시스템 메시지 추가"""
        message = Message(
            role="system",
            content=content,
            stage=self._current_stage,
        )
        self._messages.append(message)

    def _add_user_message(self, content: str) -> None:
        """사용자 메시지 추가"""
        message = Message(
            role="user",
            content=content,
            stage=self._current_stage,
        )
        self._messages.append(message)

    def _add_assistant_message(self, content: str) -> None:
        """어시스턴트 메시지 추가"""
        message = Message(
            role="assistant",
            content=content,
            stage=self._current_stage,
        )
        self._messages.append(message)

    def _get_context_messages(self, max_messages: int = 20) -> List[Dict]:
        """
        컨텍스트 메시지 조회 (토큰 제한 관리)

        Args:
            max_messages: 최대 메시지 수

        Returns:
            List[Dict]: OpenAI API 형식의 메시지 리스트
        """
        # 최근 메시지만 유지
        recent_messages = self._messages[-max_messages:]

        # 시스템 메시지는 항상 포함
        system_messages = [m for m in self._messages if m.role == "system"]
        other_messages = [m for m in recent_messages if m.role != "system"]

        # 중복 제거 후 결합
        all_messages = system_messages + other_messages

        # OpenAI API 형식으로 변환
        return [
            {"role": m.role, "content": m.content}
            for m in all_messages
        ]

    async def start_interview(self) -> str:
        """
        면접 시작

        Returns:
            str: 시작 인사말
        """
        self._current_stage = InterviewStage.GREETING

        # 템플릿 메시지 결합
        greeting = " ".join(INTERVIEW_TEMPLATES[InterviewStage.GREETING])

        # 어시스턴트 메시지로 추가
        self._add_assistant_message(greeting)

        # 콜백 호출
        if self._on_question_generated:
            self._on_question_generated(greeting)
        if self._on_stage_changed:
            self._on_stage_changed(self._current_stage)

        logger.info("Interview started")
        return greeting

    async def process_answer(
        self,
        answer: str,
        auto_advance_stage: bool = True,
    ) -> str:
        """
        지원자의 답변 처리 및 다음 질문 생성

        Args:
            answer: 지원자의 답변
            auto_advance_stage: 자동 단계 진행 여부

        Returns:
            str: 다음 질문
        """
        # 사용자 답변 추가
        self._add_user_message(answer)

        # 질문 카운트
        self._question_count += 1
        self._stage_question_count[self._current_stage] += 1

        logger.debug(f"Processing answer (stage={self._current_stage.value}, count={self._question_count})")

        # 다음 단계로 진행할지 결정
        if auto_advance_stage:
            await self._check_stage_advancement()

        # LLM으로 다음 질문 생성
        next_question = await self._generate_next_question()

        # 어시스턴트 메시지로 추가
        self._add_assistant_message(next_question)

        # 콜백 호출
        if self._on_question_generated:
            self._on_question_generated(next_question)

        return next_question

    async def _check_stage_advancement(self) -> None:
        """단계 진행 체크 및 업데이트"""
        current_count = self._stage_question_count[self._current_stage]

        # 단계별 질문 수 제한
        stage_limits = {
            InterviewStage.GREETING: 1,
            InterviewStage.SELF_INTRODUCTION: 2,
            InterviewStage.EXPERIENCE: 5,
            InterviewStage.TECHNICAL: 5,
            InterviewStage.SITUATIONAL: 3,
            InterviewStage.CLOSING: 2,
            InterviewStage.FAREWELL: 1,
        }

        # 현재 단계의 제한 확인
        if current_count >= stage_limits.get(self._current_stage, 5):
            await self._advance_stage()

    async def _advance_stage(self) -> None:
        """다음 단계로 진행"""
        stage_order = [
            InterviewStage.GREETING,
            InterviewStage.SELF_INTRODUCTION,
            InterviewStage.EXPERIENCE,
            InterviewStage.TECHNICAL,
            InterviewStage.SITUATIONAL,
            InterviewStage.CLOSING,
            InterviewStage.FAREWELL,
        ]

        current_index = stage_order.index(self._current_stage)

        if current_index < len(stage_order) - 1:
            next_stage = stage_order[current_index + 1]
            self._current_stage = next_stage

            logger.info(f"Stage advanced to: {next_stage.value}")

            # 콜백 호출
            if self._on_stage_changed:
                self._on_stage_changed(next_stage)

    async def _generate_next_question(self) -> str:
        """
        LLM으로 다음 질문 생성

        Returns:
            str: 생성된 질문
        """
        # 현재 단계 정보를 추가 프롬프트로
        stage_instruction = self._get_stage_instruction()

        # 임시 시스템 메시지
        temp_messages = self._get_context_messages()
        temp_messages.append({
            "role": "system",
            "content": stage_instruction,
        })

        try:
            # OpenAI API 호출
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=temp_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # 응답 추출
            question = response.choices[0].message.content.strip()

            logger.debug(f"Generated question: {question[:50]}...")
            return question

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")

            # 폴백: 템플릿에서 질문 가져오기
            return self._get_fallback_question()

    def _get_stage_instruction(self) -> str:
        """현재 단계에 맞는 추가 지시사항"""
        instructions = {
            InterviewStage.GREETING: """
면접을 시작하는 단계입니다. 지원자를 환영하고 자기소개를 요청하세요.
짧고 따뜻한 인사로 편안한 분위기를 만들어주세요.
""",
            InterviewStage.SELF_INTRODUCTION: """
지원자의 자기소개를 듣는 단계입니다.
학력, 경력, 기술 스택에 대해 질문하세요.
지원자의 답변을 바탕으로 후속 질문을 하세요.
""",
            InterviewStage.EXPERIENCE: """
지원자의 실무 경험을 탐색하는 단계입니다.
구체적인 프로젝트, 역할, 성과에 대해 질문하세요.
STAR 기법을 활용해 상황-과제-행동-결과를 파악하세요.
""",
            InterviewStage.TECHNICAL: """
기술 역량을 평가하는 단계입니다.
사용 기술, 도구, 방법론에 대해 질문하세요.
문제 해결 접근 방식과 학습 능력을 확인하세요.
""",
            InterviewStage.SITUATIONAL: """
상황 대처 능력을 평가하는 단계입니다.
팀워크, 의사소통, 문제 해결 경험을 질문하세요.
구체적인 예시를 요청하세요.
""",
            InterviewStage.CLOSING: """
면접을 마무리하는 단계입니다.
지원 동기와 마지막 질문 기회를 제공하세요.
긍정적이고 격려하는 마무리를 하세요.
""",
            InterviewStage.FAREWELL: """
면접을 종료하는 단계입니다.
감사 인사와 다음 단계 안내를 하세요.
따뜻하게 마무리하세요.
""",
        }

        base_instruction = instructions.get(
            self._current_stage,
            "지원자의 답변을 바탕으로 적절한 후속 질문을 하세요."
        )

        return f"""
{base_instruction}

중요:
- 한 번에 하나의 질문만 하세요
- 짧고 명확한 문장을 사용하세요 (음성 합성 최적화)
- 존댓말을 사용하세요
- 따뜻하고 전문적인 톤을 유지하세요
"""

    def _get_fallback_question(self) -> str:
        """폴백 질문 (LLM 실패 시)"""
        templates = INTERVIEW_TEMPLATES.get(self._current_stage, [])

        if templates:
            # 아직 사용하지 않은 템플릿 찾기
            used_count = self._stage_question_count[self._current_stage]
            if used_count < len(templates):
                return templates[used_count]
            else:
                return templates[-1]  # 마지막 템플릿 재사용

        return "다음 질문입니다. 답변 부탁드립니다."

    async def end_interview(self) -> str:
        """
        면접 종료

        Returns:
            str: 종료 인사
        """
        self._current_stage = InterviewStage.FAREWELL

        farewell = " ".join(INTERVIEW_TEMPLATES[InterviewStage.FAREWELL])
        self._add_assistant_message(farewell)

        logger.info("Interview ended")
        return farewell

    def get_conversation_history(self) -> List[Message]:
        """대화 히스토리 조회"""
        return self._messages.copy()

    def get_statistics(self) -> Dict:
        """면접 통계 조회"""
        return {
            "current_stage": self._current_stage.value,
            "total_questions": self._question_count,
            "stage_questions": {
                stage.value: count
                for stage, count in self._stage_question_count.items()
            },
            "total_messages": len(self._messages),
        }

    def reset(self) -> None:
        """면접 초기화"""
        self._messages = []
        self._current_stage = InterviewStage.GREETING
        self._question_count = 0
        self._stage_question_count = {stage: 0 for stage in InterviewStage}

        # 시스템 프롬프트 다시 추가
        self._add_system_message(INTERVIEWER_SYSTEM_PROMPT)

        logger.info("Interviewer reset")


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_interviewer(
    model: str = "gpt-4o",
    on_question_generated: Optional[Callable] = None,
    on_stage_changed: Optional[Callable] = None,
) -> InterviewerAgent:
    """
    면접관 에이전트 생성 헬퍼

    Args:
        model: 사용할 LLM 모델
        on_question_generated: 질문 생성 콜백
        on_stage_changed: 단계 변경 콜백

    Returns:
        InterviewerAgent: 면접관 에이전트
    """
    return InterviewerAgent(
        model=model,
        on_question_generated=on_question_generated,
        on_stage_changed=on_stage_changed,
    )


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """면접관 에이전트 테스트"""
    import asyncio

    async def test_interviewer():
        """면접관 테스트"""

        print("=" * 80)
        print("AI Interviewer Agent Test")
        print("=" * 80)

        def on_question(question: str):
            print(f"\n[면접관] {question}")

        def on_stage_change(stage: InterviewStage):
            print(f"\n>>> Stage: {stage.value.upper()} <<<\n")

        # 면접관 생성
        interviewer = create_interviewer(
            on_question_generated=on_question,
            on_stage_changed=on_stage_change,
        )

        # 면접 시작
        await interviewer.start_interview()

        # 시뮬레이션 답변
        sample_answers = [
            "안녕하세요. 저는 5년 경력의 소프트웨어 엔지니어입니다.",
            "Python과 JavaScript를 주로 사용합니다.",
            "최근에는 AI 기반 챗봇 프로젝트를 진행했습니다.",
            "백엔드 개발과 API 설계를 담당했습니다.",
            "FastAPI와 PostgreSQL을 사용했습니다.",
            "팀원들과 적극적으로 소통하며 협업했습니다.",
            "새로운 기술을 배우는 것을 좋아합니다.",
            "이 회사의 기술 스택과 문화가 잘 맞다고 생각합니다.",
        ]

        for i, answer in enumerate(sample_answers, 1):
            print(f"\n[지원자 {i}] {answer}")
            await asyncio.sleep(0.5)

            next_question = await interviewer.process_answer(answer)
            await asyncio.sleep(1)

        # 면접 종료
        print("\n" + "=" * 80)
        await interviewer.end_interview()

        # 통계
        print("\n" + "=" * 80)
        stats = interviewer.get_statistics()
        print("Interview Statistics:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Final stage: {stats['current_stage']}")
        print("\nQuestions by stage:")
        for stage, count in stats['stage_questions'].items():
            if count > 0:
                print(f"  {stage}: {count}")

    # 테스트 실행
    asyncio.run(test_interviewer())
