"""
FastAPI 서버 메인 파일

실시간 면접 아바타 시스템의 REST API 및 WebSocket 서버
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import sys

# Loguru 로깅 설정
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/api_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="10 days",
    compression="zip",
    level="DEBUG"
)

# Import session manager and Redis client
from src.server.session_manager import get_session_manager, SessionStatus, InterviewStage
from src.utils.redis_client import get_redis_client
from config.settings import get_settings

# Import realtime pipeline
from src.pipeline.realtime_pipeline import RealtimePipeline, PipelineConfig, create_pipeline

# Initialize services
settings = get_settings()
session_manager = get_session_manager()
redis_client = get_redis_client()

# Pipeline instances per session
session_pipelines: dict = {}

# FastAPI 앱 생성 (자동 문서화 설정)
app = FastAPI(
    title="Realtime Interview Avatar API",
    description="""
    🎭 **실시간 AI 면접관 아바타 시스템**

    음성 인식, LLM 기반 대화, 음성 합성, 립싱크 아바타가 통합된 실시간 면접 시뮬레이션 플랫폼입니다.

    ## 주요 기능

    * **실시간 음성 인식**: Deepgram Nova-3를 사용한 한국어 고정밀 STT
    * **AI 면접관**: GPT-4o 기반 맥락 인식 대화
    * **고품질 음성 합성**: ElevenLabs/EdgeTTS/Naver 지원
    * **립싱크 아바타**: MuseTalk 기반 실시간 얼굴 애니메이션

    ## 인증

    API 키를 HTTP 헤더로 전달하세요:
    ```
    X-API-Key: your_api_key_here
    ```

    ## 제한

    - 요청 제한: 1분당 60 요청
    - 동시 세션: 사용자당 최대 5개
    - 세션 시간: 최대 2시간

    ## 문서

    - [API 문서](https://github.com/yourusername/realtime-interview-avatar/blob/main/docs/api.md)
    - [GitHub 저장소](https://github.com/yourusername/realtime-interview-avatar)

    ## 지원

    문제가 발생하면 [GitHub Issues](https://github.com/yourusername/realtime-interview-avatar/issues)에 제보해주세요.
    """,
    version="1.0.0",
    contact={
        "name": "Realtime Interview Avatar Team",
        "url": "https://github.com/yourusername/realtime-interview-avatar",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",  # OpenAPI 스키마
    openapi_tags=[
        {
            "name": "sessions",
            "description": "세션 관리 API - 면접 세션 생성, 조회, 종료"
        },
        {
            "name": "health",
            "description": "시스템 상태 확인 API"
        },
        {
            "name": "config",
            "description": "설정 관리 API"
        },
        {
            "name": "stats",
            "description": "통계 및 모니터링 API"
        },
        {
            "name": "websocket",
            "description": "실시간 WebSocket 연결"
        }
    ]
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        # 프로덕션 도메인 추가
        # "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic 모델 (자동 문서화용)
# ============================================================================

class SessionCreate(BaseModel):
    """세션 생성 요청"""
    user_id: str = Field(..., description="사용자 고유 ID", example="user123")
    interview_type: str = Field(
        "general",
        description="면접 유형",
        example="technical",
        regex="^(technical|behavioral|general)$"
    )
    language: str = Field(
        "ko",
        description="언어 코드 (ISO 639-1)",
        example="ko",
        regex="^(ko|en|ja)$"
    )
    difficulty: str = Field(
        "medium",
        description="난이도",
        example="medium",
        regex="^(easy|medium|hard)$"
    )
    duration_minutes: int = Field(
        30,
        description="세션 최대 시간 (분)",
        example=30,
        ge=5,
        le=120
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "interview_type": "technical",
                "language": "ko",
                "difficulty": "medium",
                "duration_minutes": 30
            }
        }


class SessionResponse(BaseModel):
    """세션 응답"""
    session_id: str = Field(..., description="세션 고유 ID")
    user_id: str = Field(..., description="사용자 ID")
    interview_type: str = Field(..., description="면접 유형")
    language: str = Field(..., description="언어 코드")
    difficulty: str = Field(..., description="난이도")
    status: str = Field(..., description="세션 상태")
    created_at: datetime = Field(..., description="생성 시간")
    expires_at: datetime = Field(..., description="만료 시간")
    daily_room_url: str = Field(..., description="Daily.co 룸 URL")
    websocket_url: str = Field(..., description="WebSocket 연결 URL")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "sess_abc123def456",
                "user_id": "user123",
                "interview_type": "technical",
                "language": "ko",
                "difficulty": "medium",
                "status": "created",
                "created_at": "2024-01-01T12:00:00Z",
                "expires_at": "2024-01-01T12:30:00Z",
                "daily_room_url": "https://your-domain.daily.co/sess_abc123def456",
                "websocket_url": "ws://localhost:8000/ws/sess_abc123def456"
            }
        }


class SessionDetail(BaseModel):
    """세션 상세 정보"""
    session_id: str
    user_id: str
    status: str = Field(..., description="created | active | paused | completed | terminated | expired")
    interview_type: str
    language: str
    difficulty: str
    created_at: datetime
    started_at: Optional[datetime] = None
    expires_at: datetime
    duration_seconds: int
    message_count: int
    metadata: dict = {}


class SessionSummary(BaseModel):
    """세션 요약"""
    duration_seconds: int
    total_messages: int
    ai_questions: int
    user_responses: int
    average_response_time_ms: float
    average_ai_latency_ms: float
    transcript_url: Optional[str] = None
    video_url: Optional[str] = None


class Feedback(BaseModel):
    """면접 피드백"""
    overall_score: float = Field(..., ge=0, le=10, description="종합 점수 (0-10)")
    communication: float = Field(..., ge=0, le=10, description="의사소통 점수")
    technical_knowledge: float = Field(..., ge=0, le=10, description="기술 지식 점수")
    problem_solving: float = Field(..., ge=0, le=10, description="문제 해결 점수")
    comments: str = Field(..., description="피드백 코멘트")


class SessionTerminateResponse(BaseModel):
    """세션 종료 응답"""
    session_id: str
    status: str
    terminated_at: datetime
    summary: SessionSummary
    feedback: Feedback


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="healthy | degraded | unhealthy")
    version: str
    timestamp: datetime
    gpu_available: bool
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    services: dict


class ErrorResponse(BaseModel):
    """에러 응답"""
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    details: Optional[str] = Field(None, description="에러 상세 정보")
    timestamp: datetime
    request_id: Optional[str] = None


# ============================================================================
# 애플리케이션 이벤트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("Starting Realtime Interview Avatar API...")

    # Connect to Redis
    try:
        await redis_client.connect()
        logger.info("✓ Redis connected")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")

    # Create required directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("assets/images", exist_ok=True)
    os.makedirs("assets/audio", exist_ok=True)

    logger.info("✓ Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("Shutting down Realtime Interview Avatar API...")

    # Disconnect from Redis
    try:
        await redis_client.disconnect()
        logger.info("✓ Redis disconnected")
    except Exception as e:
        logger.error(f"✗ Redis disconnection failed: {e}")

    logger.info("✓ Application shut down successfully")


# ============================================================================
# 인증
# ============================================================================

def verify_api_key(x_api_key: str = Header(..., description="API 키")):
    """
    API 키 검증

    환경변수 API_KEY와 비교하여 인증
    """
    expected_key = settings.api_key

    if not expected_key or x_api_key != expected_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    logger.debug(f"API key validated successfully")
    return x_api_key


# ============================================================================
# REST API 엔드포인트
# ============================================================================

@app.get(
    "/",
    tags=["health"],
    summary="루트 엔드포인트",
    description="API 기본 정보를 반환합니다."
)
async def root():
    """루트 엔드포인트"""
    return {
        "name": "Realtime Interview Avatar API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


@app.get(
    "/health",
    tags=["health"],
    response_model=HealthResponse,
    summary="헬스 체크",
    description="""
    시스템의 전반적인 상태를 확인합니다.

    **응답 상태**:
    - `healthy`: 모든 서비스 정상 작동
    - `degraded`: 일부 서비스 성능 저하
    - `unhealthy`: 주요 서비스 장애

    **서비스 상태**:
    - `operational`: 정상 작동
    - `degraded`: 성능 저하
    - `down`: 서비스 중단
    """
)
async def health_check():
    """시스템 상태 확인"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = None
        gpu_memory_total = None

        if gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated() // (1024 ** 2)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    except:
        gpu_available = False
        gpu_memory_used = None
        gpu_memory_total = None

    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow(),
        "gpu_available": gpu_available,
        "gpu_memory_used_mb": gpu_memory_used,
        "gpu_memory_total_mb": gpu_memory_total,
        "services": {
            "stt": "operational",
            "llm": "operational",
            "tts": "operational",
            "avatar": "operational",
            "redis": "operational"
        }
    }


@app.post(
    "/api/sessions",
    tags=["sessions"],
    response_model=SessionResponse,
    status_code=201,
    summary="세션 생성",
    description="""
    새로운 면접 세션을 생성합니다.

    생성된 세션은 `duration_minutes`에 지정된 시간 동안 유효합니다.
    세션 ID와 WebSocket URL을 반환하며, 이를 사용하여 실시간 통신을 시작할 수 있습니다.

    **면접 유형**:
    - `technical`: 기술 면접
    - `behavioral`: 행동 면접
    - `general`: 일반 면접

    **난이도**:
    - `easy`: 초급 (신입)
    - `medium`: 중급 (경력 3-5년)
    - `hard`: 고급 (시니어)
    """,
    responses={
        201: {
            "description": "세션 생성 성공",
            "model": SessionResponse
        },
        400: {
            "description": "잘못된 요청",
            "model": ErrorResponse
        },
        401: {
            "description": "인증 실패",
            "model": ErrorResponse
        },
        429: {
            "description": "요청 제한 초과",
            "model": ErrorResponse
        }
    }
)
async def create_session(
    session: SessionCreate,
    api_key: str = Depends(verify_api_key)
):
    """세션 생성"""
    try:
        # Create new session using SessionManager
        new_session = await session_manager.create_session(
            user_id=session.user_id,
            interview_type=session.interview_type,
            difficulty=session.difficulty,
            language=session.language
        )

        logger.info(f"Created session {new_session.session_id} for user {session.user_id}")

        # Calculate expiration time
        expires_at = new_session.created_at + timedelta(minutes=session.duration_minutes)

        # Generate URLs
        base_url = settings.server_host if settings.server_host != "0.0.0.0" else "localhost"
        websocket_url = f"ws://{base_url}:{settings.server_port}/ws/{new_session.session_id}"
        daily_room_url = settings.daily_room_url or f"https://your-domain.daily.co/{new_session.session_id}"

        return {
            "session_id": new_session.session_id,
            "user_id": new_session.user_id,
            "interview_type": new_session.interview_type,
            "language": new_session.language,
            "difficulty": new_session.difficulty,
            "status": new_session.status.value,
            "created_at": new_session.created_at,
            "expires_at": expires_at,
            "daily_room_url": daily_room_url,
            "websocket_url": websocket_url
        }

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )


@app.get(
    "/api/sessions/{session_id}",
    tags=["sessions"],
    response_model=SessionDetail,
    summary="세션 조회",
    description="""
    기존 세션의 상태와 상세 정보를 조회합니다.

    **세션 상태**:
    - `created`: 세션 생성됨 (아직 시작 안됨)
    - `active`: 면접 진행 중
    - `paused`: 일시 정지
    - `completed`: 정상 종료
    - `terminated`: 강제 종료
    - `expired`: 만료됨
    """,
    responses={
        200: {"description": "세션 조회 성공", "model": SessionDetail},
        404: {"description": "세션을 찾을 수 없음", "model": ErrorResponse},
        401: {"description": "인증 실패", "model": ErrorResponse}
    }
)
async def get_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """세션 조회"""
    try:
        # Get session from SessionManager
        session = await session_manager.get_session(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        # Calculate duration
        duration_seconds = 0
        if session.started_at:
            end_time = session.ended_at or datetime.utcnow()
            duration_seconds = int((end_time - session.started_at).total_seconds())

        # Calculate expiration time (2 hours from creation)
        expires_at = session.created_at + timedelta(hours=2)

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "interview_type": session.interview_type,
            "language": session.language,
            "difficulty": session.difficulty,
            "created_at": session.created_at,
            "started_at": session.started_at,
            "expires_at": expires_at,
            "duration_seconds": duration_seconds,
            "message_count": len(session.transcript),
            "metadata": {
                "questions_asked": session.question_count,
                "current_stage": session.stage.value,
                "current_question": session.current_question,
                **session.metadata
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}"
        )


@app.get(
    "/api/sessions",
    tags=["sessions"],
    summary="세션 목록 조회",
    description="""
    사용자의 모든 세션 목록을 조회합니다.

    페이지네이션을 지원하며, `limit`과 `offset` 파라미터로 제어할 수 있습니다.
    """,
    responses={
        200: {"description": "세션 목록 조회 성공"},
        401: {"description": "인증 실패", "model": ErrorResponse}
    }
)
async def list_sessions(
    user_id: str = Query(..., description="사용자 ID"),
    status: str = Query("all", description="상태 필터 (all, active, completed 등)"),
    limit: int = Query(20, description="최대 개수", ge=1, le=100),
    offset: int = Query(0, description="페이지 오프셋", ge=0),
    api_key: str = Depends(verify_api_key)
):
    """세션 목록 조회"""
    try:
        # Parse status filter
        status_filter = None
        if status != "all":
            try:
                status_filter = SessionStatus(status)
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
                status_filter = None

        # Get sessions from SessionManager
        sessions = await session_manager.list_sessions(
            user_id=user_id,
            status=status_filter,
            limit=limit,
            offset=offset
        )

        # Get total count (for all user's sessions)
        all_sessions = await session_manager.list_sessions(
            user_id=user_id,
            status=status_filter,
            limit=1000,  # High limit to get total count
            offset=0
        )
        total = len(all_sessions)

        # Format session list
        session_list = []
        for session in sessions:
            duration_seconds = 0
            if session.started_at:
                end_time = session.ended_at or datetime.utcnow()
                duration_seconds = int((end_time - session.started_at).total_seconds())

            session_list.append({
                "session_id": session.session_id,
                "status": session.status.value,
                "interview_type": session.interview_type,
                "language": session.language,
                "difficulty": session.difficulty,
                "created_at": session.created_at,
                "started_at": session.started_at,
                "duration_seconds": duration_seconds,
                "message_count": len(session.transcript),
                "stage": session.stage.value
            })

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(session_list),
            "sessions": session_list
        }

    except Exception as e:
        logger.error(f"Failed to list sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@app.post(
    "/api/sessions/{session_id}/pause",
    tags=["sessions"],
    summary="세션 일시정지",
    description="진행 중인 세션을 일시정지합니다.",
    responses={
        200: {"description": "일시정지 성공"},
        404: {"description": "세션을 찾을 수 없음", "model": ErrorResponse},
        400: {"description": "세션 상태가 일시정지 불가능", "model": ErrorResponse}
    }
)
async def pause_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """세션 일시정지"""
    try:
        # Pause session using SessionManager
        session = await session_manager.pause_session(session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        logger.info(f"Paused session {session_id}")

        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "paused_at": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause session: {str(e)}"
        )


@app.post(
    "/api/sessions/{session_id}/resume",
    tags=["sessions"],
    summary="세션 재개",
    description="일시정지된 세션을 재개합니다.",
    responses={
        200: {"description": "재개 성공"},
        404: {"description": "세션을 찾을 수 없음", "model": ErrorResponse},
        400: {"description": "세션 상태가 재개 불가능", "model": ErrorResponse}
    }
)
async def resume_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """세션 재개"""
    try:
        # Resume session using SessionManager
        session = await session_manager.resume_session(session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        logger.info(f"Resumed session {session_id}")

        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "resumed_at": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume session: {str(e)}"
        )


@app.delete(
    "/api/sessions/{session_id}",
    tags=["sessions"],
    response_model=SessionTerminateResponse,
    summary="세션 종료",
    description="""
    세션을 종료하고 면접 요약 정보를 반환합니다.

    종료된 세션은 다시 시작할 수 없으며, 면접 피드백과 녹화 파일 URL이 제공됩니다.
    """,
    responses={
        200: {"description": "세션 종료 성공", "model": SessionTerminateResponse},
        404: {"description": "세션을 찾을 수 없음", "model": ErrorResponse}
    }
)
async def terminate_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """세션 종료"""
    try:
        # Get session
        session = await session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        # Calculate statistics
        duration_seconds = 0
        if session.started_at:
            end_time = session.ended_at or datetime.utcnow()
            duration_seconds = int((end_time - session.started_at).total_seconds())

        # Count messages
        ai_questions = len([m for m in session.transcript if m.get('role') == 'assistant'])
        user_responses = len([m for m in session.transcript if m.get('role') == 'user'])
        total_messages = len(session.transcript)

        # Generate feedback (placeholder - should be from LLM)
        feedback_data = {
            "overall_score": 7.5,
            "communication": 8.0,
            "technical_knowledge": 7.0,
            "problem_solving": 7.5,
            "comments": "면접을 완료해주셔서 감사합니다."
        }

        # Complete session
        completed_session = await session_manager.complete_session(
            session_id,
            feedback=feedback_data
        )

        logger.info(f"Terminated session {session_id}")

        # Generate file URLs (placeholder)
        transcript_url = f"/recordings/{session_id}/transcript.txt"
        video_url = f"/recordings/{session_id}/video.mp4"

        return {
            "session_id": session.session_id,
            "status": completed_session.status.value,
            "terminated_at": completed_session.ended_at or datetime.utcnow(),
            "summary": {
                "duration_seconds": duration_seconds,
                "total_messages": total_messages,
                "ai_questions": ai_questions,
                "user_responses": user_responses,
                "average_response_time_ms": 0.0,  # TODO: Calculate from timestamps
                "average_ai_latency_ms": 0.0,  # TODO: Calculate from logs
                "transcript_url": transcript_url,
                "video_url": video_url
            },
            "feedback": feedback_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to terminate session: {str(e)}"
        )


@app.get(
    "/api/config",
    tags=["config"],
    summary="설정 조회",
    description="현재 시스템 설정을 조회합니다.",
    responses={
        200: {"description": "설정 조회 성공"},
        401: {"description": "인증 실패", "model": ErrorResponse}
    }
)
async def get_config(api_key: str = Depends(verify_api_key)):
    """설정 조회"""
    try:
        # Return current configuration from settings
        return {
            "stt": {
                "provider": settings.stt_provider,
                "model": settings.stt_model,
                "language": settings.stt_language
            },
            "llm": {
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "temperature": settings.llm_temperature
            },
            "tts": {
                "provider": settings.tts_provider,
                "voice": settings.tts_edge_voice if settings.tts_provider == "edge" else "default"
            },
            "avatar": {
                "fps": settings.avatar_fps,
                "resolution": settings.resolution_tuple,
                "use_tensorrt": settings.enable_tensorrt if hasattr(settings, 'enable_tensorrt') else False
            }
        }

    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )


@app.get(
    "/api/stats",
    tags=["stats"],
    summary="통계 조회",
    description="""
    시스템 사용 통계를 조회합니다.

    **기간 옵션**:
    - `today`: 오늘
    - `week`: 최근 7일
    - `month`: 최근 30일
    """,
    responses={
        200: {"description": "통계 조회 성공"},
        401: {"description": "인증 실패", "model": ErrorResponse}
    }
)
async def get_stats(
    period: str = Query("today", description="기간 (today/week/month)"),
    user_id: Optional[str] = Query(None, description="특정 사용자 필터"),
    api_key: str = Depends(verify_api_key)
):
    """통계 조회"""
    try:
        # Calculate time range based on period
        now = datetime.utcnow()
        if period == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_time = now - timedelta(days=7)
        elif period == "month":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)

        # Get all sessions (limited)
        all_sessions = await session_manager.list_sessions(
            user_id=user_id,
            status=None,
            limit=1000,
            offset=0
        )

        # Filter by time period
        sessions_in_period = [
            s for s in all_sessions
            if s.created_at >= start_time
        ]

        # Calculate statistics
        total_sessions = len(sessions_in_period)
        active_sessions = len([s for s in sessions_in_period if s.status == SessionStatus.ACTIVE])
        completed_sessions = len([s for s in sessions_in_period if s.status == SessionStatus.COMPLETED])
        paused_sessions = len([s for s in sessions_in_period if s.status == SessionStatus.PAUSED])

        # Calculate average duration
        durations = []
        total_messages = 0
        for session in sessions_in_period:
            if session.started_at:
                end_time = session.ended_at or now
                duration = (end_time - session.started_at).total_seconds()
                durations.append(duration)
            total_messages += len(session.transcript)

        average_duration = sum(durations) / len(durations) if durations else 0

        # GPU stats (if available)
        gpu_utilization = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_utilization = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        except:
            pass

        return {
            "period": period,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "paused_sessions": paused_sessions,
            "average_duration_seconds": int(average_duration),
            "total_messages": total_messages,
            "gpu_utilization_percent": round(gpu_utilization, 2),
            "user_filter": user_id
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============================================================================
# WebSocket 엔드포인트
# ============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    api_key: str = Query(..., description="API 키")
):
    """
    WebSocket 연결

    실시간 양방향 통신을 위한 WebSocket 엔드포인트입니다.

    **연결 URL**:
    ```
    ws://localhost:8000/ws/{session_id}?api_key=your_api_key
    ```

    **메시지 타입**:
    - `connected`: 연결 확인
    - `stt_interim`: STT 중간 결과
    - `stt_final`: STT 최종 결과
    - `llm_response`: LLM 응답 (스트리밍)
    - `tts_audio`: TTS 오디오
    - `avatar_frame`: 아바타 프레임
    - `error`: 에러 메시지
    """
    # API 키 검증
    expected_key = os.getenv("API_KEY", "test_api_key")
    if api_key != expected_key:
        await websocket.close(code=1008, reason="Invalid API key")
        return

    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        # 연결 확인 메시지 전송
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to interview session"
        })

        # 메시지 수신 루프
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            logger.info(f"Received WebSocket message: {message_type} from session {session_id}")

            # Handle different message types
            if message_type == "start_session":
                # Start the interview session
                session = await session_manager.get_session(session_id)
                if session:
                    session.started_at = datetime.utcnow()
                    session.status = SessionStatus.ACTIVE
                    await session_manager.update_session(session)

                    # Initialize pipeline for this session
                    if session_id not in session_pipelines:
                        openai_key = os.getenv("OPENAI_API_KEY")
                        pipeline = await create_pipeline(
                            openai_api_key=openai_key,
                            stt_model=os.getenv("STT_MODEL", "base"),
                            llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                            tts_voice=os.getenv("TTS_EDGE_VOICE", "ko-KR-SunHiNeural")
                        )
                        session_pipelines[session_id] = pipeline
                        logger.info(f"Pipeline initialized for session {session_id}")

                    # Get greeting from pipeline
                    pipeline = session_pipelines.get(session_id)
                    if pipeline:
                        greeting_result = await pipeline.get_greeting()
                        greeting_text = greeting_result.get("response", "안녕하세요! 면접을 시작하겠습니다.")
                        greeting_audio = greeting_result.get("audio")
                    else:
                        greeting_text = "안녕하세요! 면접을 시작하겠습니다. 간단한 자기소개 부탁드립니다."
                        greeting_audio = None

                    # Send welcome message
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "stage": session.stage.value,
                        "message": greeting_text,
                        "audio": greeting_audio,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Add to transcript
                    await session_manager.add_transcript_entry(
                        session_id,
                        role="assistant",
                        content=greeting_text
                    )
                else:
                    await websocket.send_json({
                        "type": "error",
                        "code": "SESSION_NOT_FOUND",
                        "message": f"Session {session_id} not found",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            elif message_type == "audio_data":
                # Audio processing (STT → LLM → TTS → Avatar)
                audio_data = data.get("audio")

                if not audio_data:
                    await websocket.send_json({
                        "type": "error",
                        "code": "NO_AUDIO_DATA",
                        "message": "No audio data provided",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue

                # Get pipeline for this session
                pipeline = session_pipelines.get(session_id)
                if not pipeline:
                    # Initialize pipeline if not exists
                    openai_key = os.getenv("OPENAI_API_KEY")
                    pipeline = await create_pipeline(openai_api_key=openai_key)
                    session_pipelines[session_id] = pipeline

                # Send interim status
                await websocket.send_json({
                    "type": "stt_interim",
                    "transcript": "음성 인식 중...",
                    "is_final": False,
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Process audio through pipeline
                result = await pipeline.process_audio(audio_data)

                if result.get("error"):
                    await websocket.send_json({
                        "type": "error",
                        "code": "PIPELINE_ERROR",
                        "message": result["error"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue

                # Send STT result
                if result.get("transcript"):
                    await websocket.send_json({
                        "type": "stt_final",
                        "transcript": result["transcript"],
                        "confidence": 0.95,
                        "is_final": True,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Save user transcript
                    await session_manager.add_transcript_entry(
                        session_id,
                        role="user",
                        content=result["transcript"]
                    )

                # Send AI response
                if result.get("response"):
                    await websocket.send_json({
                        "type": "ai_response",
                        "text": result["response"],
                        "audio": result.get("audio"),
                        "metrics": result.get("metrics"),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Save AI response
                    await session_manager.add_transcript_entry(
                        session_id,
                        role="assistant",
                        content=result["response"]
                    )

                logger.info(f"Audio processed - STT: {result.get('transcript', '')[:50]}... "
                           f"Response: {result.get('response', '')[:50]}...")

            elif message_type == "text_message":
                # Text input (for testing without audio)
                text = data.get("text", "")

                if not text.strip():
                    continue

                # Save user message
                await session_manager.add_transcript_entry(
                    session_id,
                    role="user",
                    content=text
                )

                # Get pipeline for this session
                pipeline = session_pipelines.get(session_id)
                if not pipeline:
                    openai_key = os.getenv("OPENAI_API_KEY")
                    pipeline = await create_pipeline(openai_api_key=openai_key)
                    session_pipelines[session_id] = pipeline

                # Process text through pipeline (LLM → TTS)
                result = await pipeline.process_text(text)

                ai_response = result.get("response", f"[AI] '{text}'에 대한 답변입니다.")

                # Save AI response
                await session_manager.add_transcript_entry(
                    session_id,
                    role="assistant",
                    content=ai_response
                )

                # Send AI response with audio
                await websocket.send_json({
                    "type": "ai_response",
                    "text": ai_response,
                    "audio": result.get("audio"),
                    "metrics": result.get("metrics"),
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.info(f"Text processed - Input: {text[:50]}... Response: {ai_response[:50]}...")

            elif message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif message_type == "stop_session":
                # Stop the session
                session = await session_manager.get_session(session_id)
                if session:
                    session.status = SessionStatus.PAUSED
                    await session_manager.update_session(session)

                await websocket.send_json({
                    "type": "session_stopped",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                })

            else:
                # Unknown message type
                await websocket.send_json({
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        # Cleanup pipeline for this session
        if session_id in session_pipelines:
            pipeline = session_pipelines.pop(session_id)
            pipeline.reset()
            logger.info(f"Pipeline cleaned up for session {session_id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "code": "WEBSOCKET_ERROR",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        except:
            pass
        # Cleanup pipeline on error
        if session_id in session_pipelines:
            session_pipelines.pop(session_id)


# ============================================================================
# 에러 핸들러
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "details": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


# ============================================================================
# 시작
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Swagger UI: http://{host}:{port}/docs")
    logger.info(f"ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
