"""
Session management for interview sessions.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass, asdict
from loguru import logger

from src.utils.redis_client import get_redis_client


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class InterviewStage(str, Enum):
    """Interview stage enumeration."""
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    EXPERIENCE = "experience"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLOSING = "closing"


@dataclass
class InterviewSession:
    """Interview session data model."""
    session_id: str
    user_id: str
    interview_type: str
    difficulty: str
    language: str
    status: SessionStatus
    stage: InterviewStage
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    question_count: int = 0
    current_question: Optional[str] = None
    transcript: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.transcript is None:
            self.transcript = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['ended_at'] = self.ended_at.isoformat() if self.ended_at else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterviewSession':
        """Create instance from dictionary."""
        # Convert ISO format to datetime
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('ended_at'):
            data['ended_at'] = datetime.fromisoformat(data['ended_at'])

        # Convert string enums
        if isinstance(data.get('status'), str):
            data['status'] = SessionStatus(data['status'])
        if isinstance(data.get('stage'), str):
            data['stage'] = InterviewStage(data['stage'])

        return cls(**data)


class SessionManager:
    """
    Manages interview sessions with Redis backend.
    """

    def __init__(self):
        """Initialize session manager."""
        self.redis = get_redis_client()
        self.session_ttl = 7200  # 2 hours

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"session:{session_id}"

    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user's sessions."""
        return f"user_sessions:{user_id}"

    async def create_session(
        self,
        user_id: str,
        interview_type: str = "general",
        difficulty: str = "medium",
        language: str = "ko"
    ) -> InterviewSession:
        """
        Create new interview session.

        Args:
            user_id: User identifier
            interview_type: Type of interview (technical/behavioral/general)
            difficulty: Difficulty level (easy/medium/hard)
            language: Interview language (ko/en)

        Returns:
            Created session
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            interview_type=interview_type,
            difficulty=difficulty,
            language=language,
            status=SessionStatus.ACTIVE,
            stage=InterviewStage.GREETING,
            created_at=now,
            updated_at=now,
        )

        # Store in Redis
        session_key = self._get_session_key(session_id)
        await self.redis.set(session_key, session.to_dict(), expire=self.session_ttl)

        # Add to user's sessions
        user_sessions_key = self._get_user_sessions_key(user_id)
        user_sessions = await self.redis.get(user_sessions_key) or []
        user_sessions.append(session_id)
        await self.redis.set(user_sessions_key, user_sessions, expire=self.session_ttl * 2)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        session_key = self._get_session_key(session_id)
        data = await self.redis.get(session_key)

        if data:
            return InterviewSession.from_dict(data)
        return None

    async def update_session(self, session: InterviewSession) -> bool:
        """
        Update existing session.

        Args:
            session: Session to update

        Returns:
            True if successful
        """
        session.updated_at = datetime.utcnow()
        session_key = self._get_session_key(session.session_id)

        success = await self.redis.set(
            session_key,
            session.to_dict(),
            expire=self.session_ttl
        )

        if success:
            logger.info(f"Updated session {session.session_id}")
        return success

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        # Remove from user's sessions
        user_sessions_key = self._get_user_sessions_key(session.user_id)
        user_sessions = await self.redis.get(user_sessions_key) or []
        if session_id in user_sessions:
            user_sessions.remove(session_id)
            await self.redis.set(user_sessions_key, user_sessions)

        # Delete session
        session_key = self._get_session_key(session_id)
        result = await self.redis.delete(session_key)

        logger.info(f"Deleted session {session_id}")
        return result

    async def pause_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        Pause session.

        Args:
            session_id: Session identifier

        Returns:
            Updated session or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.PAUSED
        await self.update_session(session)

        logger.info(f"Paused session {session_id}")
        return session

    async def resume_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        Resume paused session.

        Args:
            session_id: Session identifier

        Returns:
            Updated session or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.ACTIVE
        await self.update_session(session)

        logger.info(f"Resumed session {session_id}")
        return session

    async def complete_session(
        self,
        session_id: str,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Optional[InterviewSession]:
        """
        Complete session.

        Args:
            session_id: Session identifier
            feedback: Optional feedback data

        Returns:
            Completed session or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = SessionStatus.COMPLETED
        session.ended_at = datetime.utcnow()
        if feedback:
            session.metadata['feedback'] = feedback

        await self.update_session(session)

        logger.info(f"Completed session {session_id}")
        return session

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[InterviewSession]:
        """
        List sessions with filters.

        Args:
            user_id: Filter by user ID
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of sessions
        """
        sessions = []

        if user_id:
            # Get user's sessions
            user_sessions_key = self._get_user_sessions_key(user_id)
            session_ids = await self.redis.get(user_sessions_key) or []
        else:
            # Get all sessions
            pattern = "session:*"
            keys = await self.redis.keys(pattern)
            session_ids = [k.replace("session:", "") for k in keys]

        # Fetch sessions
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                # Apply status filter
                if status and session.status != status:
                    continue
                sessions.append(session)

        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        # Apply pagination
        return sessions[offset:offset + limit]

    async def add_transcript_entry(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Add entry to session transcript.

        Args:
            session_id: Session identifier
            role: Speaker role (user/assistant)
            content: Transcript content
            timestamp: Optional timestamp

        Returns:
            True if successful
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        entry = {
            "role": role,
            "content": content,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "stage": session.stage.value
        }

        session.transcript.append(entry)
        return await self.update_session(session)

    async def update_stage(
        self,
        session_id: str,
        stage: InterviewStage
    ) -> Optional[InterviewSession]:
        """
        Update interview stage.

        Args:
            session_id: Session identifier
            stage: New interview stage

        Returns:
            Updated session or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        session.stage = stage
        await self.update_session(session)

        logger.info(f"Updated session {session_id} to stage {stage}")
        return session


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get global session manager instance (singleton).

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
