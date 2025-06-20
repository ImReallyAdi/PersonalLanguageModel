from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import os
import json

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Model(Base):
    """Table for storing trained models"""
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    vocab_size = Column(Integer, nullable=False)
    embed_dim = Column(Integer, nullable=False)
    num_heads = Column(Integer, nullable=False)
    num_layers = Column(Integer, nullable=False)
    sequence_length = Column(Integer, nullable=False)
    total_parameters = Column(Integer, nullable=False)
    model_data = Column(LargeBinary, nullable=False)  # Serialized model weights
    char_to_idx = Column(Text, nullable=False)  # JSON string
    idx_to_char = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class TrainingSession(Base):
    """Table for storing training session information"""
    __tablename__ = "training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), nullable=True)  # Links to model if training completes
    training_text_length = Column(Integer, nullable=False)
    sequence_length = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    num_epochs = Column(Integer, nullable=False)
    embed_dim = Column(Integer, nullable=False)
    num_heads = Column(Integer, nullable=False)
    num_layers = Column(Integer, nullable=False)
    final_loss = Column(Float)
    status = Column(String(50), default="started")  # started, training, completed, error
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)

class TrainingEpoch(Base):
    """Table for storing individual epoch results"""
    __tablename__ = "training_epochs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    epoch_number = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class GenerationRequest(Base):
    """Table for storing text generation requests and results"""
    __tablename__ = "generation_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), nullable=False)
    prompt = Column(Text, nullable=False)
    generated_text = Column(Text, nullable=False)
    max_length = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    top_k = Column(Integer)
    top_p = Column(Float)
    generation_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingData(Base):
    """Table for storing training datasets"""
    __tablename__ = "training_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    text_content = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)
    unique_chars = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Database functions
def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get database session for direct use"""
    return SessionLocal()

# Model operations
def save_model_to_db(session: Session, model_data: dict) -> str:
    """Save a trained model to the database"""
    db_model = Model(
        name=model_data.get('name', f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        description=model_data.get('description', ''),
        vocab_size=model_data['vocab_size'],
        embed_dim=model_data['embed_dim'],
        num_heads=model_data['num_heads'],
        num_layers=model_data['num_layers'],
        sequence_length=model_data['sequence_length'],
        total_parameters=model_data['total_parameters'],
        model_data=model_data['model_weights'],  # Binary data
        char_to_idx=json.dumps(model_data['char_to_idx']),
        idx_to_char=json.dumps(model_data['idx_to_char'])
    )
    
    session.add(db_model)
    session.commit()
    session.refresh(db_model)
    return str(db_model.id)

def load_model_from_db(session: Session, model_id: str = None) -> dict:
    """Load a model from the database"""
    if model_id:
        model = session.query(Model).filter(Model.id == model_id, Model.is_active == True).first()
    else:
        # Get the most recent model
        model = session.query(Model).filter(Model.is_active == True).order_by(Model.created_at.desc()).first()
    
    if not model:
        return None
    
    return {
        'id': str(model.id),
        'name': model.name,
        'description': model.description,
        'vocab_size': model.vocab_size,
        'embed_dim': model.embed_dim,
        'num_heads': model.num_heads,
        'num_layers': model.num_layers,
        'sequence_length': model.sequence_length,
        'total_parameters': model.total_parameters,
        'model_weights': model.model_data,
        'char_to_idx': json.loads(model.char_to_idx),
        'idx_to_char': json.loads(model.idx_to_char),
        'created_at': model.created_at
    }

def list_models(session: Session, limit: int = 10) -> list:
    """List available models"""
    models = session.query(Model).filter(Model.is_active == True).order_by(Model.created_at.desc()).limit(limit).all()
    
    return [{
        'id': str(model.id),
        'name': model.name,
        'description': model.description,
        'vocab_size': model.vocab_size,
        'total_parameters': model.total_parameters,
        'created_at': model.created_at
    } for model in models]

# Training session operations
def create_training_session(session: Session, config: dict) -> str:
    """Create a new training session record"""
    training_session = TrainingSession(
        training_text_length=config['training_text_length'],
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    session.add(training_session)
    session.commit()
    session.refresh(training_session)
    return str(training_session.id)

def update_training_session(session: Session, session_id: str, updates: dict):
    """Update training session with results"""
    training_session = session.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if training_session:
        for key, value in updates.items():
            setattr(training_session, key, value)
        session.commit()

def log_training_epoch(session: Session, session_id: str, epoch: int, loss: float, lr: float):
    """Log individual epoch results"""
    epoch_record = TrainingEpoch(
        session_id=session_id,
        epoch_number=epoch,
        loss=loss,
        learning_rate=lr
    )
    session.add(epoch_record)
    session.commit()

# Generation operations
def log_generation_request(session: Session, request_data: dict) -> str:
    """Log a text generation request"""
    generation = GenerationRequest(
        model_id=request_data['model_id'],
        prompt=request_data['prompt'],
        generated_text=request_data['generated_text'],
        max_length=request_data['max_length'],
        temperature=request_data['temperature'],
        top_k=request_data.get('top_k'),
        top_p=request_data.get('top_p'),
        generation_time_ms=request_data.get('generation_time_ms')
    )
    
    session.add(generation)
    session.commit()
    session.refresh(generation)
    return str(generation.id)

# Training data operations
def save_training_data(session: Session, name: str, text_content: str, description: str = '') -> str:
    """Save training dataset to database"""
    unique_chars = len(set(text_content))
    
    training_data = TrainingData(
        name=name,
        description=description,
        text_content=text_content,
        text_length=len(text_content),
        unique_chars=unique_chars
    )
    
    session.add(training_data)
    session.commit()
    session.refresh(training_data)
    return str(training_data.id)

def list_training_data(session: Session, limit: int = 10) -> list:
    """List available training datasets"""
    datasets = session.query(TrainingData).filter(TrainingData.is_active == True).order_by(TrainingData.created_at.desc()).limit(limit).all()
    
    return [{
        'id': str(dataset.id),
        'name': dataset.name,
        'description': dataset.description,
        'text_length': dataset.text_length,
        'unique_chars': dataset.unique_chars,
        'created_at': dataset.created_at
    } for dataset in datasets]

def get_training_data(session: Session, data_id: str) -> dict:
    """Get training dataset by ID"""
    dataset = session.query(TrainingData).filter(TrainingData.id == data_id, TrainingData.is_active == True).first()
    
    if not dataset:
        return None
    
    return {
        'id': str(dataset.id),
        'name': dataset.name,
        'description': dataset.description,
        'text_content': dataset.text_content,
        'text_length': dataset.text_length,
        'unique_chars': dataset.unique_chars,
        'created_at': dataset.created_at
    }

# Analytics functions
def get_training_statistics(session: Session, days: int = 30) -> dict:
    """Get training statistics for the last N days"""
    from sqlalchemy import func
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Total sessions
    total_sessions = session.query(TrainingSession).filter(TrainingSession.started_at >= cutoff_date).count()
    
    # Completed sessions
    completed_sessions = session.query(TrainingSession).filter(
        TrainingSession.started_at >= cutoff_date,
        TrainingSession.status == 'completed'
    ).count()
    
    # Average training time
    avg_duration = session.query(func.avg(TrainingSession.duration_seconds)).filter(
        TrainingSession.started_at >= cutoff_date,
        TrainingSession.status == 'completed'
    ).scalar()
    
    # Total models created
    total_models = session.query(Model).filter(Model.created_at >= cutoff_date).count()
    
    return {
        'total_sessions': total_sessions,
        'completed_sessions': completed_sessions,
        'success_rate': completed_sessions / total_sessions if total_sessions > 0 else 0,
        'average_duration_seconds': int(avg_duration) if avg_duration else 0,
        'total_models': total_models
    }

def get_generation_statistics(session: Session, days: int = 30) -> dict:
    """Get generation statistics for the last N days"""
    from sqlalchemy import func
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Total generations
    total_generations = session.query(GenerationRequest).filter(GenerationRequest.created_at >= cutoff_date).count()
    
    # Average generation time
    avg_time = session.query(func.avg(GenerationRequest.generation_time_ms)).filter(
        GenerationRequest.created_at >= cutoff_date
    ).scalar()
    
    # Average text length
    avg_length = session.query(func.avg(func.length(GenerationRequest.generated_text))).filter(
        GenerationRequest.created_at >= cutoff_date
    ).scalar()
    
    return {
        'total_generations': total_generations,
        'average_time_ms': int(avg_time) if avg_time else 0,
        'average_text_length': int(avg_length) if avg_length else 0
    }

# Initialize database
def init_database():
    """Initialize the database with tables"""
    try:
        create_tables()
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False