# from datetime import datetime
# from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, relationship
# import json

# Base = declarative_base()

# class User(Base):
#     __tablename__ = 'users'
#     id = Column(Integer, primary_key=True)
#     session_id = Column(String(100), unique=True)
#     conversations = relationship("Conversation", back_populates="user")

# class Conversation(Base):
#     __tablename__ = 'conversations'
#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, ForeignKey('users.id'))
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     user = relationship("User", back_populates="conversations")
#     messages = relationship("Message", back_populates="conversation")

# class Message(Base):
#     __tablename__ = 'messages'
#     id = Column(Integer, primary_key=True)
#     conversation_id = Column(Integer, ForeignKey('conversations.id'))
#     role = Column(String(50))  # 'user' or 'assistant'
#     content = Column(Text)
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     conversation = relationship("Conversation", back_populates="messages")

# class MemoryStore:
#     def __init__(self, database_url="sqlite:///chat_history.db"):
#         self.engine = create_engine(database_url)
#         Base.metadata.create_all(self.engine)
#         Session = sessionmaker(bind=self.engine)
#         self.session = Session()

#     def get_or_create_user(self, session_id):
#         user = self.session.query(User).filter_by(session_id=session_id).first()
#         if not user:
#             user = User(session_id=session_id)
#             self.session.add(user)
#             self.session.commit()
#         return user

#     def store_message(self, session_id, role, content):
#         user = self.get_or_create_user(session_id)
        
#         # Get or create current conversation
#         conversation = (self.session.query(Conversation)
#                       .filter_by(user_id=user.id)
#                       .order_by(Conversation.timestamp.desc())
#                       .first())
                      
#         if not conversation:
#             conversation = Conversation(user_id=user.id)
#             self.session.add(conversation)
        
#         message = Message(
#             conversation_id=conversation.id,
#             role=role,
#             content=content
#         )
#         self.session.add(message)
#         self.session.commit()

#     def get_recent_history(self, session_id, limit=5):
#         user = self.get_or_create_user(session_id)
#         messages = (self.session.query(Message)
#                    .join(Conversation)
#                    .filter(Conversation.user_id == user.id)
#                    .order_by(Message.timestamp.desc())
#                    .limit(limit)
#                    .all())
#         return [(msg.role, msg.content) for msg in reversed(messages)]

#     def get_relevant_history(self, session_id, query, limit=3):
#         """
#         Get relevant historical messages based on query similarity
#         This is a simple implementation - could be enhanced with embeddings
#         """
#         user = self.get_or_create_user(session_id)
#         messages = (self.session.query(Message)
#                    .join(Conversation)
#                    .filter(Conversation.user_id == user.id)
#                    .filter(Message.content.ilike(f"%{query}%"))
#                    .order_by(Message.timestamp.desc())
#                    .limit(limit)
#                    .all())
#         return [(msg.role, msg.content) for msg in messages]



from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from typing import List, Tuple, Optional
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

class MemoryStore:
    def __init__(self, database_url: str = "sqlite:///chat_history.db"):
        """Initialize the memory store with database connection."""
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionMaker = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionMaker()

    def get_or_create_user(self, session_id: str, db_session: Optional[Session] = None) -> User:
        """Get existing user or create new one if not exists."""
        session = db_session or self._get_session()
        try:
            user = session.query(User).filter_by(session_id=session_id).first()
            if not user:
                user = User(session_id=session_id)
                session.add(user)
                session.commit()
            return user
        finally:
            if not db_session:
                session.close()

    def get_or_create_conversation(self, session_id: str, db_session: Optional[Session] = None) -> Conversation:
        """Get current conversation or create new one."""
        session = db_session or self._get_session()
        try:
            user = self.get_or_create_user(session_id, session)
            conversation = (
                session.query(Conversation)
                .filter_by(user_id=user.id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )
            
            if not conversation:
                conversation = Conversation(user_id=user.id)
                session.add(conversation)
                session.commit()
            
            return conversation
        finally:
            if not db_session:
                session.close()

    def store_message(self, session_id: str, role: str, content: str) -> None:
        """Store a new message in the current conversation."""
        session = self._get_session()
        try:
            conversation = self.get_or_create_conversation(session_id, session)
            
            message = Message(
                conversation_id=conversation.id,
                role=role,
                content=content,
                timestamp=datetime.utcnow()
            )
            session.add(message)
            session.commit()
        finally:
            session.close()

    def get_recent_history(self, session_id: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Get recent message history for a session."""
        session = self._get_session()
        try:
            user = self.get_or_create_user(session_id, session)
            messages = (
                session.query(Message)
                .join(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Message.timestamp.desc())
                .limit(limit)
                .all()
            )
            
            # Return messages in chronological order
            return [(msg.role, msg.content) for msg in reversed(messages)]
        finally:
            session.close()

    def get_relevant_history(self, session_id: str, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        """Get relevant historical messages based on query similarity."""
        session = self._get_session()
        try:
            user = self.get_or_create_user(session_id, session)
            
            # Split query into keywords for better matching
            keywords = query.lower().split()
            messages = []
            
            # Get all messages from user's conversations
            all_messages = (
                session.query(Message)
                .join(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Message.timestamp.desc())
                .all()
            )
            
            # Score messages based on keyword matches
            scored_messages = []
            for msg in all_messages:
                score = sum(1 for keyword in keywords if keyword in msg.content.lower())
                if score > 0:
                    scored_messages.append((score, msg))
            
            # Sort by score and get top matches
            scored_messages.sort(key=lambda x: (-x[0], -x[1].timestamp.timestamp()))
            relevant_messages = [msg for _, msg in scored_messages[:limit]]
            
            return [(msg.role, msg.content) for msg in relevant_messages]
        finally:
            session.close()

    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> str:
        """Get formatted conversation context for the LLM."""
        messages = self.get_recent_history(session_id, max_messages)
        if not messages:
            return ""
        
        context = "Previous conversation:\n"
        for role, content in messages:
            context += f"{role.capitalize()}: {content}\n"
        return context.strip()

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        session = self._get_session()
        try:
            user = self.get_or_create_user(session_id, session)
            session.query(Conversation).filter_by(user_id=user.id).delete()
            session.commit()
        finally:
            session.close()