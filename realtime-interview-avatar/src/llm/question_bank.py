"""
면접 질문 데이터베이스
구조화된 면접 질문 관리 및 선택
"""

import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


# =============================================================================
# 질문 카테고리
# =============================================================================

class QuestionCategory(Enum):
    """질문 카테고리"""
    INTRODUCTION = "introduction"      # 자기소개, 지원동기
    EXPERIENCE = "experience"          # 이전 경험, 프로젝트
    TECHNICAL = "technical"            # 기술 관련
    BEHAVIORAL = "behavioral"          # 상황 대처, 팀워크
    CULTURE_FIT = "culture_fit"        # 회사 문화, 가치관
    CLOSING = "closing"                # 질문 있으신가요?, 마무리


# =============================================================================
# 질문 데이터 클래스
# =============================================================================

@dataclass
class Question:
    """면접 질문"""
    id: str                              # 질문 ID
    text: str                            # 질문 내용
    category: QuestionCategory           # 카테고리
    difficulty: int = 3                  # 난이도 (1-5)
    follow_ups: List[str] = field(default_factory=list)  # 후속 질문들
    evaluation_criteria: List[str] = field(default_factory=list)  # 평가 기준
    tags: List[str] = field(default_factory=list)  # 태그 (Python, Backend, etc.)


    def __repr__(self):
        return f"Question(id={self.id}, category={self.category.value}, difficulty={self.difficulty})"


# =============================================================================
# 면접 질문 데이터베이스
# =============================================================================

DEFAULT_QUESTIONS = [
    # =======================================================================
    # INTRODUCTION (자기소개, 지원동기)
    # =======================================================================
    Question(
        id="intro_001",
        text="간단한 자기소개 부탁드립니다.",
        category=QuestionCategory.INTRODUCTION,
        difficulty=1,
        follow_ups=[
            "주요 기술 스택은 무엇인가요?",
            "어떤 프로젝트를 주로 하셨나요?",
        ],
        evaluation_criteria=[
            "명확하고 간결한 소개",
            "관련 경험 언급",
            "의사소통 능력",
        ],
    ),

    Question(
        id="intro_002",
        text="우리 회사에 지원하신 동기가 무엇인가요?",
        category=QuestionCategory.INTRODUCTION,
        difficulty=2,
        follow_ups=[
            "우리 회사에 대해 알고 계신 것이 있나요?",
            "이 직무에서 무엇을 기대하시나요?",
        ],
        evaluation_criteria=[
            "회사에 대한 이해도",
            "지원 동기의 진정성",
            "장기적 비전",
        ],
    ),

    Question(
        id="intro_003",
        text="본인의 강점과 약점에 대해 말씀해 주세요.",
        category=QuestionCategory.INTRODUCTION,
        difficulty=2,
        follow_ups=[
            "약점을 개선하기 위해 어떤 노력을 하고 계신가요?",
            "그 강점을 프로젝트에서 어떻게 활용하셨나요?",
        ],
        evaluation_criteria=[
            "자기 인식",
            "개선 의지",
            "구체적인 예시",
        ],
    ),

    # =======================================================================
    # EXPERIENCE (이전 경험, 프로젝트)
    # =======================================================================
    Question(
        id="exp_001",
        text="최근에 진행하신 프로젝트에 대해 설명해 주시겠어요?",
        category=QuestionCategory.EXPERIENCE,
        difficulty=2,
        follow_ups=[
            "그 프로젝트에서 어떤 역할을 맡으셨나요?",
            "어떤 기술을 사용하셨나요?",
            "프로젝트의 성과는 어떠했나요?",
        ],
        evaluation_criteria=[
            "프로젝트 설명 명확성",
            "기술적 이해도",
            "역할과 기여도",
        ],
    ),

    Question(
        id="exp_002",
        text="프로젝트 중 가장 어려웠던 기술적 문제는 무엇이었나요?",
        category=QuestionCategory.EXPERIENCE,
        difficulty=3,
        follow_ups=[
            "어떻게 해결하셨나요?",
            "왜 그런 접근 방식을 선택하셨나요?",
            "다시 한다면 다르게 하실 건가요?",
        ],
        evaluation_criteria=[
            "문제 해결 능력",
            "기술적 깊이",
            "의사결정 과정",
            "회고 및 학습",
        ],
        tags=["problem_solving"],
    ),

    Question(
        id="exp_003",
        text="가장 자랑스러운 프로젝트는 무엇인가요?",
        category=QuestionCategory.EXPERIENCE,
        difficulty=2,
        follow_ups=[
            "왜 그 프로젝트가 특별한가요?",
            "어떤 영향을 미쳤나요?",
        ],
        evaluation_criteria=[
            "성취 의식",
            "프로젝트 임팩트 이해",
            "열정",
        ],
    ),

    Question(
        id="exp_004",
        text="실패했던 프로젝트나 경험이 있나요?",
        category=QuestionCategory.EXPERIENCE,
        difficulty=3,
        follow_ups=[
            "무엇이 잘못되었나요?",
            "그 경험에서 무엇을 배우셨나요?",
            "이후에 어떻게 개선하셨나요?",
        ],
        evaluation_criteria=[
            "실패 수용 능력",
            "학습 능력",
            "개선 의지",
        ],
    ),

    # =======================================================================
    # TECHNICAL (기술 관련)
    # =======================================================================
    Question(
        id="tech_001",
        text="주로 사용하시는 프로그래밍 언어와 그 이유는 무엇인가요?",
        category=QuestionCategory.TECHNICAL,
        difficulty=2,
        follow_ups=[
            "다른 언어와 비교했을 때 장단점은 무엇인가요?",
            "최근에 배운 언어가 있나요?",
        ],
        evaluation_criteria=[
            "기술 선택의 합리성",
            "언어 이해도",
            "학습 의지",
        ],
        tags=["programming", "backend"],
    ),

    Question(
        id="tech_002",
        text="코드 품질을 유지하기 위해 어떤 방법을 사용하시나요?",
        category=QuestionCategory.TECHNICAL,
        difficulty=3,
        follow_ups=[
            "테스트는 어떻게 작성하시나요?",
            "코드 리뷰는 어떻게 진행하시나요?",
        ],
        evaluation_criteria=[
            "코드 품질 의식",
            "테스트 이해도",
            "협업 프로세스",
        ],
        tags=["best_practices"],
    ),

    Question(
        id="tech_003",
        text="최근에 배운 새로운 기술이나 프레임워크가 있나요?",
        category=QuestionCategory.TECHNICAL,
        difficulty=2,
        follow_ups=[
            "어떻게 학습하셨나요?",
            "실무에 어떻게 적용하셨나요?",
        ],
        evaluation_criteria=[
            "학습 능력",
            "기술 호기심",
            "실용적 적용",
        ],
    ),

    Question(
        id="tech_004",
        text="데이터베이스 설계 경험에 대해 말씀해 주세요.",
        category=QuestionCategory.TECHNICAL,
        difficulty=3,
        follow_ups=[
            "SQL과 NoSQL 중 어떤 것을 선호하시나요? 왜죠?",
            "성능 최적화를 위해 어떤 전략을 사용하셨나요?",
        ],
        evaluation_criteria=[
            "데이터베이스 이해도",
            "설계 능력",
            "성능 최적화 경험",
        ],
        tags=["database", "backend"],
    ),

    Question(
        id="tech_005",
        text="API 설계 시 중요하게 생각하는 원칙은 무엇인가요?",
        category=QuestionCategory.TECHNICAL,
        difficulty=3,
        follow_ups=[
            "RESTful API 설계 경험이 있으신가요?",
            "버전 관리는 어떻게 하시나요?",
        ],
        evaluation_criteria=[
            "API 설계 이해도",
            "RESTful 원칙 이해",
            "실무 경험",
        ],
        tags=["api", "backend"],
    ),

    # =======================================================================
    # BEHAVIORAL (상황 대처, 팀워크)
    # =======================================================================
    Question(
        id="behav_001",
        text="팀원과 의견이 충돌했던 경험이 있나요? 어떻게 해결하셨나요?",
        category=QuestionCategory.BEHAVIORAL,
        difficulty=3,
        follow_ups=[
            "그 과정에서 배운 점은 무엇인가요?",
            "다시 비슷한 상황이 온다면 어떻게 하시겠어요?",
        ],
        evaluation_criteria=[
            "갈등 해결 능력",
            "의사소통 능력",
            "협업 태도",
        ],
    ),

    Question(
        id="behav_002",
        text="촉박한 마감 기한을 맞췄던 경험을 말씀해 주세요.",
        category=QuestionCategory.BEHAVIORAL,
        difficulty=3,
        follow_ups=[
            "어떻게 우선순위를 정하셨나요?",
            "스트레스를 어떻게 관리하셨나요?",
        ],
        evaluation_criteria=[
            "시간 관리 능력",
            "우선순위 설정",
            "스트레스 대처",
        ],
    ),

    Question(
        id="behav_003",
        text="팀에서 리더 역할을 맡았던 경험이 있나요?",
        category=QuestionCategory.BEHAVIORAL,
        difficulty=3,
        follow_ups=[
            "리더로서 어떤 점이 어려웠나요?",
            "팀원들과 어떻게 소통하셨나요?",
        ],
        evaluation_criteria=[
            "리더십",
            "팀 관리 능력",
            "의사결정 능력",
        ],
    ),

    Question(
        id="behav_004",
        text="새로운 팀에 합류했을 때 어떻게 적응하시나요?",
        category=QuestionCategory.BEHAVIORAL,
        difficulty=2,
        follow_ups=[
            "팀 문화를 파악하는 방법은?",
            "관계를 구축하는 전략은?",
        ],
        evaluation_criteria=[
            "적응력",
            "사회성",
            "학습 능력",
        ],
    ),

    Question(
        id="behav_005",
        text="업무 중 실수를 했을 때 어떻게 대처하시나요?",
        category=QuestionCategory.BEHAVIORAL,
        difficulty=3,
        follow_ups=[
            "최근에 실수한 경험이 있나요?",
            "어떻게 해결하셨나요?",
        ],
        evaluation_criteria=[
            "책임감",
            "문제 해결",
            "정직성",
        ],
    ),

    # =======================================================================
    # CULTURE_FIT (회사 문화, 가치관)
    # =======================================================================
    Question(
        id="culture_001",
        text="일과 삶의 균형에 대해 어떻게 생각하시나요?",
        category=QuestionCategory.CULTURE_FIT,
        difficulty=2,
        follow_ups=[
            "워라밸을 위해 어떤 노력을 하시나요?",
        ],
        evaluation_criteria=[
            "가치관",
            "자기 관리",
        ],
    ),

    Question(
        id="culture_002",
        text="이상적인 업무 환경은 어떤 모습인가요?",
        category=QuestionCategory.CULTURE_FIT,
        difficulty=2,
        follow_ups=[
            "왜 그런 환경을 선호하시나요?",
            "우리 회사가 그런 환경을 제공한다고 생각하시나요?",
        ],
        evaluation_criteria=[
            "회사와의 적합성",
            "기대치",
        ],
    ),

    Question(
        id="culture_003",
        text="5년 후 본인의 모습은 어떨 것 같나요?",
        category=QuestionCategory.CULTURE_FIT,
        difficulty=2,
        follow_ups=[
            "그 목표를 위해 무엇을 하고 계신가요?",
        ],
        evaluation_criteria=[
            "커리어 목표",
            "장기적 비전",
            "성장 의지",
        ],
    ),

    # =======================================================================
    # CLOSING (마무리)
    # =======================================================================
    Question(
        id="closing_001",
        text="마지막으로 하고 싶으신 말씀이 있으신가요?",
        category=QuestionCategory.CLOSING,
        difficulty=1,
        follow_ups=[],
        evaluation_criteria=[
            "종합적 인상",
            "의욕",
        ],
    ),

    Question(
        id="closing_002",
        text="궁금한 점이나 질문하고 싶으신 것이 있으신가요?",
        category=QuestionCategory.CLOSING,
        difficulty=1,
        follow_ups=[],
        evaluation_criteria=[
            "관심도",
            "준비성",
            "질문의 질",
        ],
    ),
]


# =============================================================================
# 질문 뱅크
# =============================================================================

class QuestionBank:
    """
    면접 질문 데이터베이스

    카테고리별로 질문을 관리하고, 면접 진행에 따라
    적절한 질문을 선택합니다.
    """

    def __init__(self, questions: Optional[List[Question]] = None):
        """
        Args:
            questions: 질문 리스트 (None이면 기본 질문 사용)
        """
        self.questions = questions or DEFAULT_QUESTIONS.copy()
        self._used_question_ids: Set[str] = set()

        logger.info(f"QuestionBank initialized with {len(self.questions)} questions")

    def add_question(self, question: Question) -> None:
        """
        질문 추가

        Args:
            question: 추가할 질문
        """
        self.questions.append(question)
        logger.debug(f"Question added: {question.id}")

    def add_questions(self, questions: List[Question]) -> None:
        """
        여러 질문 추가

        Args:
            questions: 추가할 질문 리스트
        """
        self.questions.extend(questions)
        logger.debug(f"{len(questions)} questions added")

    def get_by_category(
        self,
        category: QuestionCategory,
        exclude_used: bool = True,
    ) -> List[Question]:
        """
        카테고리별 질문 조회

        Args:
            category: 질문 카테고리
            exclude_used: 이미 사용한 질문 제외

        Returns:
            List[Question]: 질문 리스트
        """
        questions = [q for q in self.questions if q.category == category]

        if exclude_used:
            questions = [q for q in questions if q.id not in self._used_question_ids]

        return questions

    def get_by_difficulty(
        self,
        min_difficulty: int = 1,
        max_difficulty: int = 5,
        category: Optional[QuestionCategory] = None,
        exclude_used: bool = True,
    ) -> List[Question]:
        """
        난이도별 질문 조회

        Args:
            min_difficulty: 최소 난이도
            max_difficulty: 최대 난이도
            category: 카테고리 필터 (선택)
            exclude_used: 이미 사용한 질문 제외

        Returns:
            List[Question]: 질문 리스트
        """
        questions = [
            q for q in self.questions
            if min_difficulty <= q.difficulty <= max_difficulty
        ]

        if category:
            questions = [q for q in questions if q.category == category]

        if exclude_used:
            questions = [q for q in questions if q.id not in self._used_question_ids]

        return questions

    def get_by_tags(
        self,
        tags: List[str],
        exclude_used: bool = True,
    ) -> List[Question]:
        """
        태그별 질문 조회

        Args:
            tags: 태그 리스트
            exclude_used: 이미 사용한 질문 제외

        Returns:
            List[Question]: 질문 리스트
        """
        questions = [
            q for q in self.questions
            if any(tag in q.tags for tag in tags)
        ]

        if exclude_used:
            questions = [q for q in questions if q.id not in self._used_question_ids]

        return questions

    def select_question(
        self,
        category: QuestionCategory,
        difficulty: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Question]:
        """
        조건에 맞는 질문 하나 선택

        Args:
            category: 질문 카테고리
            difficulty: 원하는 난이도 (선택)
            tags: 원하는 태그 (선택)

        Returns:
            Question: 선택된 질문 (없으면 None)
        """
        # 카테고리로 필터링
        questions = self.get_by_category(category, exclude_used=True)

        # 난이도 필터링
        if difficulty:
            questions = [q for q in questions if q.difficulty == difficulty]

        # 태그 필터링
        if tags:
            questions = [
                q for q in questions
                if any(tag in q.tags for tag in tags)
            ]

        # 랜덤 선택
        if questions:
            question = random.choice(questions)
            self.mark_as_used(question.id)
            return question

        return None

    def select_progressive_questions(
        self,
        category: QuestionCategory,
        count: int = 3,
    ) -> List[Question]:
        """
        난이도가 점진적으로 증가하는 질문들 선택

        Args:
            category: 질문 카테고리
            count: 선택할 질문 수

        Returns:
            List[Question]: 선택된 질문 리스트
        """
        questions = self.get_by_category(category, exclude_used=True)

        if not questions:
            return []

        # 난이도순으로 정렬
        questions.sort(key=lambda q: q.difficulty)

        # 균등하게 분배하여 선택
        selected = []
        step = len(questions) // count

        for i in range(count):
            idx = min(i * step, len(questions) - 1)
            question = questions[idx]
            selected.append(question)
            self.mark_as_used(question.id)

        return selected

    def mark_as_used(self, question_id: str) -> None:
        """
        질문을 사용됨으로 표시

        Args:
            question_id: 질문 ID
        """
        self._used_question_ids.add(question_id)
        logger.debug(f"Question marked as used: {question_id}")

    def reset_used(self) -> None:
        """사용된 질문 기록 초기화"""
        self._used_question_ids.clear()
        logger.debug("Used questions reset")

    def get_statistics(self) -> Dict:
        """
        질문 뱅크 통계

        Returns:
            Dict: 통계 정보
        """
        category_counts = {}
        for category in QuestionCategory:
            count = len(self.get_by_category(category, exclude_used=False))
            category_counts[category.value] = count

        return {
            "total_questions": len(self.questions),
            "used_questions": len(self._used_question_ids),
            "available_questions": len(self.questions) - len(self._used_question_ids),
            "by_category": category_counts,
        }


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_question_bank(custom_questions: Optional[List[Question]] = None) -> QuestionBank:
    """
    질문 뱅크 생성 헬퍼

    Args:
        custom_questions: 커스텀 질문 (기본 질문에 추가됨)

    Returns:
        QuestionBank: 질문 뱅크 인스턴스
    """
    bank = QuestionBank()

    if custom_questions:
        bank.add_questions(custom_questions)

    return bank


# =============================================================================
# 테스트/예시 코드
# =============================================================================

if __name__ == "__main__":
    """질문 뱅크 테스트"""

    print("=" * 80)
    print("Question Bank Test")
    print("=" * 80)

    # 질문 뱅크 생성
    bank = create_question_bank()

    # 통계
    stats = bank.get_statistics()
    print(f"\nTotal Questions: {stats['total_questions']}")
    print("\nQuestions by Category:")
    for category, count in stats['by_category'].items():
        print(f"  {category}: {count}")

    # 카테고리별 질문 선택
    print("\n" + "=" * 80)
    print("Sample Questions by Category")
    print("=" * 80)

    for category in QuestionCategory:
        question = bank.select_question(category)
        if question:
            print(f"\n[{category.value.upper()}]")
            print(f"Q: {question.text}")
            print(f"   Difficulty: {'★' * question.difficulty}")
            if question.follow_ups:
                print(f"   Follow-ups: {len(question.follow_ups)}")

    # 점진적 질문 선택
    print("\n" + "=" * 80)
    print("Progressive Questions (TECHNICAL)")
    print("=" * 80)

    bank.reset_used()
    progressive = bank.select_progressive_questions(QuestionCategory.TECHNICAL, count=3)

    for i, q in enumerate(progressive, 1):
        print(f"\n{i}. [{q.difficulty}★] {q.text}")

    # 통계 업데이트
    print("\n" + "=" * 80)
    stats = bank.get_statistics()
    print(f"Used: {stats['used_questions']} / {stats['total_questions']}")
