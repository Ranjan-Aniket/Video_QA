# Smart Evidence Pipeline - Architecture Documentation

## System Overview

The Smart Evidence Pipeline is an **audio-first, adversarial video Q&A generation system** that uses AI to create challenging questions for testing multimodal language models. The system combines speech transcription, visual analysis, and strategic frame extraction to generate high-quality adversarial test cases.

---

## Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND (React + Vite)                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Video Upload │  │   Smart      │  │  Evidence    │  │  Settings    │           │
│  │    Page      │  │  Pipeline    │  │   Review     │  │    Page      │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                 │                      │
│         │    ┌────────────┴────────────┐    │                 │                      │
│         └────►  React Query (TanStack) ◄────┴─────────────────┘                      │
│              │  - Data Fetching        │                                             │
│              │  - Caching              │                                             │
│              │  - Real-time Polling    │                                             │
│              └────────────┬────────────┘                                             │
│                           │                                                           │
└───────────────────────────┼───────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            │
┌───────────────────────────▼───────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI + Python)                                   │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              API ENDPOINTS                                       │  │
│  ├─────────────────────────────────────────────────────────────────────────────────┤  │
│  │  /api/upload/video          - Upload video files                                │  │
│  │  /api/smart-pipeline/run    - Trigger processing pipeline                       │  │
│  │  /api/smart-pipeline/status - Get processing progress                           │  │
│  │  /api/smart-pipeline/audio  - Get audio analysis results                        │  │
│  │  /api/smart-pipeline/frames - Get extracted frames                              │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                                │
│                                        ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                    ADVERSARIAL SMART PIPELINE (Core Engine)                      │  │
│  ├─────────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                                   │  │
│  │  PHASE 1: Audio Analysis                                                         │  │
│  │  ├─ Extract audio from video (ffmpeg)                                            │  │
│  │  ├─ Transcribe with Whisper (large-v3)                                           │  │
│  │  └─ Speaker diarization + timestamps                                             │  │
│  │                        │                                                          │  │
│  │                        ▼                                                          │  │
│  │  PHASE 2: Adversarial Opportunity Mining                                         │  │
│  │  ├─ GPT-4 analyzes transcript                                                    │  │
│  │  ├─ Identify temporal markers ("at 2:34", "later", "before")                    │  │
│  │  ├─ Find ambiguous references ("this", "that object")                            │  │
│  │  ├─ Detect counting opportunities                                                │  │
│  │  ├─ Mark sequential events                                                       │  │
│  │  └─ Generate premium keyframe timestamps                                         │  │
│  │                        │                                                          │  │
│  │                        ▼                                                          │  │
│  │  PHASE 3: Smart Frame Extraction                                                 │  │
│  │  ├─ Premium Frames: Extract at identified keyframes                              │  │
│  │  ├─ Template Frames: Strategic moments (start, 25%, 50%, 75%, end)              │  │
│  │  ├─ Bulk Frames: Every 5 seconds for coverage                                    │  │
│  │  └─ Save frames with metadata                                                    │  │
│  │                        │                                                          │  │
│  │                        ▼                                                          │  │
│  │  PHASE 4: Evidence Extraction                                                    │  │
│  │  ├─ GPT-4o Vision: Analyze premium frames                                        │  │
│  │  ├─ Extract visual evidence with timestamps                                      │  │
│  │  ├─ Combine with audio transcript                                                │  │
│  │  └─ Generate evidence database                                                   │  │
│  │                        │                                                          │  │
│  │                        ▼                                                          │  │
│  │  PHASE 5: Adversarial Question Generation                                        │  │
│  │  ├─ Template Questions (20): Time-based, counting, sequence                      │  │
│  │  ├─ AI-Generated Questions (7): Context-specific from GPT-4                      │  │
│  │  ├─ Cross-Validated Questions (3): Multi-model consensus                         │  │
│  │  └─ Total: 30 adversarial questions                                              │  │
│  │                        │                                                          │  │
│  │                        ▼                                                          │  │
│  │  PHASE 6: Gemini Adversarial Testing (Optional)                                  │  │
│  │  ├─ Test questions against Gemini                                                │  │
│  │  ├─ Identify failure cases                                                       │  │
│  │  ├─ Validate question quality                                                    │  │
│  │  └─ Generate test report                                                         │  │
│  │                                                                                   │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                                │
│                                        ▼                                                │
└────────────────────────────────────────┼────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
┌─────────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   DATABASE (SQLite)     │  │  FILE STORAGE       │  │  EXTERNAL APIS      │
├─────────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│                         │  │                     │  │                     │
│ Videos Table            │  │ uploads/            │  │ OpenAI API          │
│ ├─ video_id            │  │ ├─ Original videos  │  │ ├─ Whisper large-v3 │
│ ├─ pipeline_stage      │  │ └─ Metadata         │  │ ├─ GPT-4o Vision    │
│ ├─ status              │  │                     │  │ └─ GPT-4            │
│ └─ created_at          │  │ outputs/            │  │                     │
│                         │  │ └─ {video_id}/      │  │ Google Gemini API   │
│ Questions Table         │  │    ├─ frames/       │  │ └─ gemini-1.5-pro   │
│ ├─ question_id         │  │    ├─ audio/        │  │                     │
│ ├─ question_text       │  │    ├─ analysis/     │  │                     │
│ └─ evidence_ids        │  │    └─ results/      │  │                     │
│                         │  │                     │  │                     │
│ Evidence Table          │  │ logs/               │  │                     │
│ ├─ evidence_id         │  │ └─ master.log       │  │                     │
│ ├─ timestamp           │  │                     │  │                     │
│ ├─ content             │  │                     │  │                     │
│ └─ confidence          │  │                     │  │                     │
│                         │  │                     │  │                     │
└─────────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## Key Data Flows

### 1. Video Upload Flow
```
User → Upload Page → FastAPI → Save to uploads/ → Create DB record → Return video_id
```

### 2. Pipeline Processing Flow
```
Trigger → Background Task → Phase 1-6 → Update DB → Save outputs → Complete
```

### 3. Progress Monitoring Flow
```
Frontend polls (3s) → GET /status → Return progress % + current phase → Display to user
```

### 4. Results Retrieval Flow
```
Frontend → GET /audio, /frames, /questions → Fetch from outputs/ → Return JSON → Display
```

### 5. Evidence Review Flow
```
Review Page → GET /evidence → Load from DB → Approve/Reject → Update DB → Save changes
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TypeScript, Vite, TanStack Query, Tailwind CSS |
| **Backend** | FastAPI, Python 3.9+, Pydantic |
| **Database** | SQLite 3 |
| **Video Processing** | ffmpeg, OpenCV |
| **AI Models** | Whisper (large-v3), GPT-4o, GPT-4, Gemini 1.5 Pro |
| **Storage** | Local filesystem (uploads/, outputs/) |
| **Logging** | Python logging, Master Logger |

---

## Core Components

### Frontend Components

#### 1. Video Upload Page (`/upload`)
- Drag-and-drop video upload
- Upload progress tracking
- Title and description metadata
- Auto-start pipeline option
- List of previously uploaded videos

#### 2. Smart Pipeline Page (`/smart-pipeline`)
- Real-time pipeline monitoring
- Progress bar with phase tracking
- Auto-resume on page refresh
- Tabbed interface for results:
  - Overview
  - Audio Analysis
  - Adversarial Opportunities
  - Frame Extraction
  - Questions
  - Gemini Testing

#### 3. Evidence Review Page (`/evidence`)
- Evidence approval workflow
- Visual frame display
- Timestamp tracking
- Confidence scoring

#### 4. Settings Page (`/settings`)
- API key configuration
- Processing settings
- Database management

### Backend Components

#### 1. API Router (`backend/api/endpoints/`)
- `video_upload.router` - Video upload endpoints
- `smart_pipeline_router` - Pipeline orchestration

#### 2. Processing Pipeline (`processing/`)
- `audio_analysis.py` - Whisper transcription + diarization
- `smart_pipeline.py` - Main pipeline orchestration
- `frame_extraction.py` - Smart frame extraction
- `opportunity_mining.py` - GPT-4 adversarial analysis

#### 3. Database Operations (`database/`)
- `schema.py` - SQLAlchemy models
- `operations.py` - CRUD operations
- `migrate_evidence_tables.py` - Schema migrations

---

## Real-time Features

### Polling & Updates
- **Status Polling:** Frontend polls status every 3 seconds
- **Video List:** Updates every 5 seconds
- **Progress Tracking:** Real-time progress bar (0-100%)

### State Management
- **Auto-resume:** Page refresh detects ongoing pipeline and resumes monitoring
- **Loading States:** Spinners for all data fetches
- **Error Handling:** Graceful 404 handling for incomplete phases

### User Experience
- Toast notifications for success/error events
- Loading skeletons for better perceived performance
- Optimistic UI updates

---

## Processing Pipeline Details

### Phase 1: Audio Analysis
**Duration:** ~30-60 seconds per minute of video

**Process:**
1. Extract audio using ffmpeg with quality filters
2. Transcribe with Whisper large-v3 model
3. Generate word-level timestamps
4. Perform speaker diarization
5. Save transcript with metadata

**Output:** `{video_id}/audio/transcript.json`

### Phase 2: Adversarial Opportunity Mining
**Duration:** ~10-20 seconds

**Process:**
1. GPT-4 analyzes full transcript
2. Identifies adversarial patterns:
   - Temporal markers ("at 2:34", "later")
   - Ambiguous references ("this", "that")
   - Counting opportunities
   - Sequential events
   - Context-rich moments
3. Generates premium keyframe timestamps

**Output:** `{video_id}/analysis/opportunities.json`

### Phase 3: Smart Frame Extraction
**Duration:** ~5-10 seconds

**Strategy:**
- **Premium Frames:** Extract at GPT-4 identified keyframes
- **Template Frames:** 0%, 25%, 50%, 75%, 100% marks
- **Bulk Frames:** Every 5 seconds for full coverage

**Output:** `{video_id}/frames/`
- `premium/` - High-value frames
- `template/` - Strategic frames
- `bulk/` - Coverage frames

### Phase 4: Evidence Extraction
**Duration:** ~2-3 seconds per premium frame

**Process:**
1. GPT-4o Vision analyzes premium frames
2. Extracts visual evidence with descriptions
3. Links to transcript timestamps
4. Calculates confidence scores

**Output:** Database records in `evidence` table

### Phase 5: Question Generation
**Duration:** ~30-40 seconds

**Types:**
- **Template Questions (20):** Rule-based adversarial questions
- **AI Questions (7):** GPT-4 generated context-specific
- **Cross-Validated (3):** Multi-model consensus questions

**Output:** `{video_id}/questions/questions.json`

### Phase 6: Gemini Testing (Optional)
**Duration:** ~10-15 seconds

**Process:**
1. Submit questions to Gemini 1.5 Pro
2. Compare answers to ground truth
3. Identify failure cases
4. Generate quality metrics

**Output:** `{video_id}/results/gemini_results.json`

---

## Database Schema

### Videos Table
```sql
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    video_id VARCHAR(50) UNIQUE NOT NULL,
    video_url TEXT NOT NULL,
    video_name VARCHAR(255),
    pipeline_stage VARCHAR(50),
    status VARCHAR(20),
    created_at DATETIME,
    -- Additional tracking fields...
)
```

### Questions Table
```sql
CREATE TABLE questions (
    id INTEGER PRIMARY KEY,
    question_id VARCHAR(50) UNIQUE,
    video_id VARCHAR(50),
    question_text TEXT,
    question_type VARCHAR(50),
    difficulty VARCHAR(20),
    evidence_ids TEXT,
    -- Additional metadata...
)
```

### Evidence Table
```sql
CREATE TABLE evidence (
    id INTEGER PRIMARY KEY,
    evidence_id VARCHAR(50) UNIQUE,
    video_id VARCHAR(50),
    timestamp FLOAT,
    content TEXT,
    evidence_type VARCHAR(50),
    confidence FLOAT,
    -- Additional fields...
)
```

---

## API Endpoints

### Video Upload
- `POST /api/upload/video` - Upload video file
- `GET /api/upload/list` - List all uploaded videos
- `DELETE /api/upload/{video_id}` - Delete video

### Smart Pipeline
- `POST /api/smart-pipeline/run` - Start pipeline processing
- `GET /api/smart-pipeline/status/{video_id}` - Get progress
- `GET /api/smart-pipeline/audio/{video_id}` - Get audio analysis
- `GET /api/smart-pipeline/opportunities/{video_id}` - Get adversarial opportunities
- `GET /api/smart-pipeline/frames/{video_id}` - Get frame extraction results
- `GET /api/smart-pipeline/questions/{video_id}` - Get generated questions
- `GET /api/smart-pipeline/gemini-results/{video_id}` - Get Gemini test results
- `GET /api/smart-pipeline/transcript/{video_id}` - Get full transcript
- `DELETE /api/smart-pipeline/{video_id}` - Delete all pipeline results

### Evidence Review
- `GET /api/evidence/review` - Get evidence for review
- `POST /api/evidence/approve` - Approve evidence
- `POST /api/evidence/reject` - Reject evidence

---

## File Structure

```
Gemini_QA/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── api/                # API client
│   │   ├── components/         # Reusable components
│   │   ├── pages/              # Page components
│   │   ├── store/              # Zustand state management
│   │   └── hooks/              # Custom hooks (WebSocket, etc.)
│   └── package.json
│
├── backend/                     # FastAPI backend
│   ├── api/
│   │   └── endpoints/          # API route handlers
│   └── main.py                 # FastAPI app
│
├── processing/                  # Core processing logic
│   ├── audio_analysis.py       # Whisper + diarization
│   ├── smart_pipeline.py       # Pipeline orchestrator
│   ├── frame_extraction.py     # Frame extraction
│   └── opportunity_mining.py   # GPT-4 analysis
│
├── database/                    # Database layer
│   ├── schema.py               # SQLAlchemy models
│   └── operations.py           # DB operations
│
├── config/                      # Configuration
│   └── settings.py             # Environment settings
│
├── uploads/                     # Uploaded videos
├── outputs/                     # Processing outputs
│   └── {video_id}/
│       ├── frames/
│       ├── audio/
│       ├── analysis/
│       └── results/
│
├── logs/                        # Application logs
│   └── master.log
│
└── test.db                      # SQLite database
```

---

## Configuration

### Environment Variables (.env)
```bash
# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Processing Settings
MAX_WORKERS=4
USE_GPU=false
AUTO_RETRY=true

# Database
DATABASE_URL=sqlite:///test.db

# Logging
LOG_LEVEL=DEBUG
```

---

## Performance Optimizations

### Smart Frame Extraction
- **91.6% efficiency gain** vs analyzing every frame
- Premium frames target high-value moments
- Bulk frames provide coverage safety net

### Caching
- React Query caches API responses
- 3-second polling minimizes redundant requests
- SQLite database for fast metadata access

### Background Processing
- FastAPI BackgroundTasks for async processing
- Non-blocking pipeline execution
- Real-time progress updates

---

## Error Handling

### Backend
- HTTPException for proper status codes (404, 500, etc.)
- Detailed error logging in master.log
- Graceful failure recovery

### Frontend
- Error boundaries for component failures
- Toast notifications for user feedback
- Loading states prevent confusion
- 404s handled gracefully (data not ready yet)

---

## Deployment

### Development
```bash
# Start backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend
cd frontend && npm run dev
```

### Production
```bash
# Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
npm run build && npx vite preview
```

---

## Future Enhancements

1. **Multi-video Batch Processing**
2. **Advanced Caching with Redis**
3. **WebSocket for Real-time Updates**
4. **Cloud Storage Integration**
5. **Advanced Analytics Dashboard**
6. **Question Difficulty Scoring**
7. **Export to Multiple Formats**

---

## Key Benefits

✅ **Audio-First Approach** - Transcribe once, analyze thoroughly
✅ **Smart Frame Selection** - 91.6% fewer API calls vs naive approach
✅ **Adversarial Focus** - Identifies model weaknesses
✅ **Multi-Model Testing** - Cross-validates with Gemini
✅ **Real-time Monitoring** - Live progress tracking
✅ **Production Ready** - Error handling, logging, persistence

---

**Last Updated:** 2025-11-18
**Version:** 3.0.0
**Architecture:** Smart Evidence Pipeline - Adversarial Q&A Generation
