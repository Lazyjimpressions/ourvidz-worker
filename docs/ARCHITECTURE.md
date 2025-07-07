# OurVidz.com - Technical Architecture

**Last Updated:** July 6, 2025 at 10:11 AM CST  
**System:** Dual Worker Architecture on RTX 6000 ADA (48GB VRAM)  
**Deployment:** Production on Lovable (https://ourvidz.lovable.app/)  
**Status:** ✅ All 10 Job Types Available - Testing Phase

---

## **System Architecture Overview**

### **High-Level Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Workers       │
│   (React/TS)    │◄──►│   (Supabase)    │◄──►│   (RunPod)      │
│   Lovable.app   │    │   Production    │    │   RTX 6000 ADA  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Components │    │   Edge Functions│    │   Dual Workers  │
│   - Job Selection│    │   - queue-job   │    │   - SDXL Worker │
│   - Asset Display│    │   - job-callback│    │   - WAN Worker  │
│   - Workspace   │    │   - Auth        │    │   - Qwen Worker │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Dual Worker System (Current)**
```yaml
Worker Architecture:
  SDXL Worker:
    Queue: sdxl_queue (2s polling)
    Job Types: sdxl_image_fast, sdxl_image_high
    Performance: 3-8 seconds per image
    Output: 6-image batch (array of URLs)
    VRAM Usage: 6.6GB loaded, 10.5GB peak
    Status: ✅ Fully operational

  WAN Worker:
    Queue: wan_queue (5s polling)
    Job Types: 8 types (4 standard + 4 enhanced)
    Performance: 67-294 seconds
    Output: Single file (image/video URL)
    VRAM Usage: 15-30GB peak
    Enhancement: Qwen 7B (14.6s) - Currently disabled
    Status: ✅ Operational (enhanced jobs working but quality issues)

  Qwen Worker (Planned):
    Purpose: Prompt enhancement and storytelling
    Model: Qwen 7B (NSFW-capable)
    Integration: Same server as WAN/SDXL
    Storage: Temp storage for speed, persistence for stability
    Status: 🚧 Planning phase
```

---

## **Frontend Architecture**

### **Technology Stack**
```yaml
Core Framework:
  React: 18.3.1 with TypeScript 5.5.3
  Build Tool: Vite 5.4.1
  Styling: Tailwind CSS 3.4.11 + shadcn/ui components
  State Management: React Context + React Query 5.56.2
  Routing: React Router 6.26.2

Key Libraries:
  @tanstack/react-query: Server state management
  @supabase/supabase-js: 2.50.0 Backend integration
  lucide-react: 0.462.0 Icons
  sonner: 1.5.0 Toast notifications
  react-hook-form: 7.53.0 Form handling
  zod: 3.23.8 Schema validation

Deployment:
  Platform: Lovable
  URL: https://ourvidz.lovable.app/
  Status: ✅ Production deployed
```

### **Component Architecture**
```yaml
Layout Components:
  OurVidzDashboardLayout: Main dashboard layout
  PortalLayout: Modal/overlay layouts
  AuthHeader: Authentication header

Core Features:
  Workspace: Main generation interface
  Library: Asset management and display
  Dashboard: User overview and navigation

Generation Components:
  FastImageGenerator: SDXL fast generation (6-image batch)
  HighImageGenerator: SDXL high-quality generation (6-image batch)
  FastVideoGenerator: WAN fast video generation (single file)
  HighVideoGenerator: WAN high-quality video generation (single file)
  Enhanced Generators: Qwen-enhanced versions (currently disabled)

Asset Management:
  MediaGrid: Asset display grid
  AssetCard: Individual asset display
  AssetFilters: Filtering and sorting
  AssetTableView: Table view of assets
  LibraryImportModal: Import from library to workspace
```

### **State Management**
```yaml
Context Providers:
  AuthContext: User authentication state
  GenerationContext: Generation state and progress

Custom Hooks:
  useGeneration: Generation workflow management
  useGenerationStatus: Real-time status updates
  useJobQueue: Job queue management
  useAssets: Asset management with React Query
  useWorkspace: Workspace state management
  useGenerationWorkspace: Auto-add generated content
  useProject: Project-level state
```

---

## **Backend Architecture**

### **Supabase Services**
```yaml
Database (PostgreSQL 15.8):
  Tables:
    - jobs: Job tracking and status
    - images: Generated image metadata
    - videos: Generated video metadata
    - profiles: User information
    - projects: Project management
    - scenes: Scene management
    - characters: Character management
    - user_roles: Role-based access control
    - usage_logs: Usage tracking

  RLS Policies:
    - Users can only access their own data
    - Admins have full access to all data
    - Job status updates restricted to workers
    - Asset access controlled by ownership

Storage:
  Buckets (12 Total):
    - sdxl_image_fast: SDXL fast images (5MB limit)
    - sdxl_image_high: SDXL high-quality images (10MB limit)
    - image_fast: WAN fast images (5MB limit)
    - image_high: WAN high-quality images (10MB limit)
    - video_fast: WAN fast videos (50MB limit)
    - video_high: WAN high-quality videos (200MB limit)
    - image7b_fast_enhanced: Enhanced fast images (20MB limit)
    - image7b_high_enhanced: Enhanced high-quality images (20MB limit)
    - video7b_fast_enhanced: Enhanced fast videos (100MB limit)
    - video7b_high_enhanced: Enhanced high-quality videos (100MB limit)
    - videos: Public video storage (no limit)
    - system_assets: Public system assets (5MB limit)

  Policies:
    - Private access (authenticated users only)
    - File size limits: 5MB-200MB depending on bucket
    - Allowed types: PNG for images, MP4 for videos

Edge Functions:
  queue-job: Job creation and queue routing (JWT verification enabled)
  job-callback: Job completion handling (JWT verification disabled)
  generate-admin-image: Admin image generation (Admin bypass)
```

### **Redis Queue System**
```yaml
Provider: Upstash Redis (REST API)
Queues:
  sdxl_queue: SDXL job processing (2s polling)
  wan_queue: WAN job processing (5s polling)

Job Structure:
  id: Unique job identifier
  type: Job type (e.g., sdxl_image_fast)
  prompt: User input prompt
  config: Generation parameters
  user_id: User identifier
  created_at: Timestamp
  status: pending/processing/completed/failed
```

---

## **Worker System Architecture**

### **SDXL Worker Implementation**
```python
# sdxl_worker.py
class SDXLWorker:
    def __init__(self):
        self.model_path = "/workspace/models/sdxl-lustify"
        self.device = "cuda"
        self.pipe = None  # Lazy loading
        
    def load_model(self):
        # Load SDXL model with NSFW capabilities
        # 6.5GB model size, 6.6GB VRAM usage
        
    def generate(self, prompt, config):
        # Generate 6 images per job (BATCH GENERATION)
        # Performance: 3-8 seconds per image
        # Output: Array of 6 image URLs
        # Storage: Multiple files in sdxl bucket
```

### **WAN Worker Implementation**
```python
# wan_worker.py
class WANWorker:
    def __init__(self):
        self.wan_path = "/workspace/models/wan2.1-t2v-1.3b"
        self.device = "cuda"
        
    def generate_video(self, prompt, config):
        # WAN 2.1 video generation
        # Performance: 67-280 seconds
        # Output: Single video file URL
        # Storage: Single file in video bucket
        
    def generate_image(self, prompt, config):
        # WAN 2.1 image generation
        # Performance: 67-90 seconds
        # Output: Single image file URL
        # Storage: Single file in image bucket
```

### **Qwen Worker (Planned)**
```python
# qwen_worker.py (Future Implementation)
class QwenWorker:
    def __init__(self):
        self.model_path = "/workspace/models/qwen-7b"
        self.device = "cuda"
        
    def enhance_prompt(self, prompt, model_type):
        # Qwen 7B prompt enhancement
        # Purpose: NSFW content enhancement
        # Integration: Pre-processing for WAN/SDXL
        
    def generate_storyboard(self, script):
        # Storyboarding functionality
        # Future feature for video production
```

---

## **Data Flow Architecture**

### **Job Processing Flow**
```yaml
1. User Input:
   Frontend: User enters prompt and selects job type
   Validation: Client-side validation of input

2. Job Creation:
   Edge Function: queue-job.ts creates job record
   Database: Job stored in jobs table
   Queue: Job added to appropriate Redis queue

3. Worker Processing:
   Worker: Polls queue for new jobs
   SDXL Jobs: Generate 6 images, return array of URLs
   WAN Jobs: Generate single file, return single URL
   Storage: Content uploaded to Supabase bucket

4. Completion:
   Callback: job-callback.ts updates job status
   Database: Asset metadata stored in images/videos table
   Frontend: Real-time status updates via polling
```

### **Asset Management Flow**
```yaml
1. Generation Complete:
   SDXL: Uploads 6 images to bucket, returns array of URLs
   WAN: Uploads single file to bucket, returns single URL
   Metadata: Stores asset information in database
   Status: Updates job status to completed

2. Frontend Display:
   React Query: Fetches assets from database
   Grid Display: MediaGrid shows assets
   SDXL Display: Shows 6 images with selection options
   WAN Display: Shows single image/video

3. User Interaction:
   Preview: AssetPreviewModal shows full-size content
   Download: Direct download from Supabase bucket
   Delete: Asset deletion with confirmation
```

---

## **Performance Architecture**

### **GPU Memory Management**
```yaml
RTX 6000 ADA (48GB VRAM):
  SDXL Worker:
    Model Load: 6.6GB
    Generation Peak: 10.5GB
    Cleanup: 0GB (perfect cleanup)
    
  WAN Worker:
    Model Load: ~15GB
    Generation Peak: 15-30GB
    Qwen Enhancement: 8-12GB (currently disabled)
    
  Concurrent Operation:
    Total Peak: ~35GB
    Available: 13GB headroom
    Strategy: Sequential loading/unloading
```

### **Optimization Strategies**
```yaml
Model Loading:
  Lazy Loading: Models loaded only when needed
  Persistence: All models stored on network volume
  Caching: Models remain in memory during session
  
Queue Management:
  Polling Intervals: SDXL (2s), WAN (5s)
  Job Batching: SDXL generates 6 images per job
  Priority Handling: Fast jobs processed first
  
Storage Optimization:
  Compression: Videos compressed for storage
  Cleanup: Temporary files removed after processing
  Bucket Organization: Separate buckets by job type
```

---

## **Security Architecture**

### **Authentication & Authorization**
```yaml
Supabase Auth:
  Provider: Supabase Auth with email/password
  Session Management: JWT tokens
  RLS Policies: Row-level security on all tables
  
Access Control:
  User Isolation: Users can only access their own data
  Admin Access: Full access to all data and functions
  Job Security: Jobs tied to authenticated users
  Asset Protection: Assets protected by user ownership
```

### **API Security**
```yaml
Edge Functions:
  Authentication: Required for all job operations
  Rate Limiting: Built into Supabase
  Input Validation: Server-side validation
  
Worker Security:
  Environment Variables: Sensitive data in env vars
  Network Isolation: Workers in secure environment
  Model Access: Models stored in secure volume
```

---

## **Monitoring & Observability**

### **System Monitoring**
```yaml
Worker Health:
  Process Monitoring: Dual orchestrator tracks worker processes
  Auto-Restart: Failed workers automatically restarted
  Status Reporting: Real-time status updates
  
Performance Tracking:
  Job Timing: Generation time tracking per job type
  Success Rates: Job completion success rates
  Resource Usage: GPU memory and utilization monitoring
  
Error Handling:
  Job Failures: Failed jobs logged with error details
  Retry Logic: Automatic retry for transient failures
  User Notifications: Error messages sent to users
```

### **Logging Strategy**
```yaml
Application Logs:
  Frontend: Console logs for debugging
  Backend: Supabase logs for API calls
  Workers: File-based logging for job processing
  
Monitoring Tools:
  Supabase Dashboard: Database and API monitoring
  RunPod Dashboard: GPU and resource monitoring
  Custom Metrics: Job success rates and performance
```

---

## **Deployment Architecture**

### **Environment Configuration**
```yaml
Development:
  Frontend: Local development server
  Backend: Supabase development project
  Workers: Local testing environment
  
Production:
  Frontend: Lovable deployment (https://ourvidz.lovable.app/)
  Backend: Supabase production project
  Workers: RunPod production environment
  
Environment Variables:
  SUPABASE_URL: Backend connection
  SUPABASE_ANON_KEY: Frontend authentication
  SUPABASE_SERVICE_KEY: Backend operations
  UPSTASH_REDIS_REST_URL: Queue connection
  UPSTASH_REDIS_REST_TOKEN: Queue authentication
```

### **Scaling Considerations**
```yaml
Current Capacity:
  Single GPU: RTX 6000 ADA (48GB)
  Concurrent Jobs: 1 SDXL + 1 WAN simultaneously
  Queue Capacity: Unlimited (Redis-based)
  
Future Scaling:
  Multi-GPU: Additional RunPod instances
  Load Balancing: Multiple worker instances
  Auto-Scaling: Dynamic worker allocation based on queue depth
```

---

## **Integration Points**

### **External Services**
```yaml
Supabase:
  Database: PostgreSQL with RLS
  Storage: Object storage with policies
  Auth: User authentication and sessions
  Edge Functions: Serverless API endpoints
  
Upstash Redis:
  Queue Management: Job queuing and processing
  REST API: HTTP-based queue operations
  Persistence: Queue data persistence
  
RunPod:
  GPU Infrastructure: RTX 6000 ADA instances
  Network Storage: Persistent model storage
  Container Management: Worker deployment
  
Lovable:
  Frontend Deployment: Production hosting
  Domain: ourvidz.lovable.app
```

### **API Endpoints**
```yaml
Frontend → Backend:
  POST /api/queue-job: Create new generation job
  GET /api/jobs: Fetch user's jobs
  GET /api/assets: Fetch user's assets
  DELETE /api/assets/:id: Delete asset
  
Backend → Workers:
  Redis Queues: Job distribution
  HTTP Callbacks: Job completion notifications
  
Workers → Backend:
  POST /api/job-callback: Update job status
  Storage Upload: Upload generated content
```

---

## **Development Workflow**

### **Code Organization**
```yaml
Frontend Structure:
  src/
    components/: Reusable UI components
    pages/: Route components
    hooks/: Custom React hooks
    contexts/: React context providers
    lib/: Utility functions and services
    types/: TypeScript type definitions
    
Backend Structure:
  supabase/
    functions/: Edge functions
    migrations/: Database migrations
    config.toml: Supabase configuration
    
Worker Structure:
  workers/: Python worker scripts
  models/: Model storage and configuration
  requirements.txt: Python dependencies
```

### **Testing Strategy**
```yaml
Frontend Testing:
  Unit Tests: Component testing with Jest
  Integration Tests: API integration testing
  E2E Tests: Full workflow testing
  
Backend Testing:
  Edge Functions: Function testing
  Database: Migration testing
  API: Endpoint testing
  
Worker Testing:
  Model Loading: Model initialization testing
  Generation: Output quality testing
  Performance: Timing and resource usage testing
```

---

## **Current Testing Status**

### **✅ Successfully Tested Job Types**
```yaml
SDXL Jobs:
  sdxl_image_fast: ✅ Working (6-image batch)
  sdxl_image_high: ✅ Working (6-image batch)

WAN Jobs:
  image_fast: ✅ Working (single file)
  video7b_fast_enhanced: ✅ Working (single file)
  video7b_high_enhanced: ✅ Working (single file)

Pending Testing:
  image_high: ❌ Not tested
  video_fast: ❌ Not tested
  video_high: ❌ Not tested
  image7b_fast_enhanced: ❌ Not tested
  image7b_high_enhanced: ❌ Not tested
```

### **Known Issues**
```yaml
Enhanced Video Quality:
  Issue: Enhanced video generation working but quality not great
  Problem: Adult/NSFW enhancement doesn't work well out of the box
  Impact: Adds 60 seconds to video generation
  Solution: Planning to use Qwen for prompt enhancement instead

File Storage Mapping:
  Issue: Job types to storage bucket mapping complexity
  Problem: URL generation and file presentation on frontend
  Impact: SDXL returns 6 images vs WAN returns single file
  Solution: Proper array handling for SDXL, single URL for WAN
```

---

## **Edge Function Implementation Details**

### **Queue-Job Edge Function (`queue-job.ts`)**
**Purpose**: Job creation and queue routing with standardized parameter handling
- **Authentication**: JWT verification required
- **Job Type Validation**: All 10 job types supported
- **Queue Routing**: SDXL → sdxl_queue, WAN → wan_queue
- **Parameter Standardization**: Consistent field names across all workers

**Supported Job Types**:
```yaml
SDXL Jobs (Fast Image Generation):
  sdxl_image_fast: 6-image batch, PNG, 1024x1024
  sdxl_image_high: 6-image batch, PNG, 1024x1024

WAN Jobs (Video + Image Generation):
  image_fast: Single image, PNG, 480x832
  image_high: Single image, PNG, 480x832
  video_fast: Single video, MP4, 480x832, 5s duration
  video_high: Single video, MP4, 480x832, 6s duration
  image7b_fast_enhanced: Enhanced image, PNG, 480x832
  image7b_high_enhanced: Enhanced image, PNG, 480x832
  video7b_fast_enhanced: Enhanced video, MP4, 480x832, 5s duration
  video7b_high_enhanced: Enhanced video, MP4, 480x832, 6s duration
```

**Job Payload Structure**:
```typescript
// Standardized payload for all workers
{
  id: string,              // Database job ID
  type: string,            // Job type (e.g., sdxl_image_fast)
  prompt: string,          // User input prompt
  user_id: string,         // User identifier
  config: {                // Generation configuration
    size: string,          // Image/video size
    sample_steps: number,  // Generation steps
    sample_guide_scale: number, // Guidance scale
    frame_num: number,     // Frame count (video only)
    enhance_prompt: boolean, // Qwen enhancement flag
    content_type: string,  // 'image' or 'video'
    file_extension: string // 'png' or 'mp4'
  },
  created_at: string,      // ISO timestamp
  video_id?: string,       // Optional video ID
  image_id?: string,       // Optional image ID
  metadata?: object        // Additional metadata
}
```

**Critical Fixes Applied**:
```yaml
Negative Prompt Handling:
  SDXL Jobs: ✅ Generate negative prompts (supported)
  WAN Jobs: ❌ No negative prompts (WAN 2.1 doesn't support --negative_prompt)
  
Parameter Consistency:
  Field Names: ✅ Standardized (id, type, prompt, user_id)
  Config Structure: ✅ Consistent across all job types
  Queue Routing: ✅ Proper routing based on job type
  
Enhanced Job Support:
  Job Type Parsing: ✅ Handles all 10 job types
  Queue Assignment: ✅ Enhanced jobs route to wan_queue
  Configuration: ✅ Proper config for enhanced features
```

### **Job Callback Edge Function (`job-callback.ts`)**
**Purpose**: Central callback handler for OurVidz AI content generation workers
- **SDXL Workers**: Handle batch image generation (6 images per job)
- **WAN Workers**: Handle single image/video generation with AI enhancement
- **Status Management**: Update jobs, images, and videos tables
- **File Path Handling**: Normalize and store generated content URLs

**Callback Parameters**:
```typescript
// Required Parameters
{
  job_id: string,          // Database job ID (snake_case format)
  status: 'processing' | 'completed' | 'failed',  // Job status
}

// Optional Parameters
{
  assets?: string[],       // Array of file URLs (SDXL batch or WAN single)
  error_message?: string,  // Error details for failed jobs
  enhanced_prompt?: string, // AI-enhanced prompt (for enhanced jobs)
}

// Backward Compatibility (Both formats supported)
{
  jobId?: string,          // camelCase (legacy format)
  job_id?: string,         // snake_case (new format)
  assets?: string[],       // snake_case (new array format)
  filePath?: string,       // camelCase (legacy SDXL)
  outputUrl?: string,      // camelCase (legacy WAN)
  errorMessage?: string,   // camelCase (legacy)
  error_message?: string,  // snake_case (new format)
}
```

**Database Updates**:
```typescript
// Jobs Table Updates
{
  status: string,                    // 'processing' | 'completed' | 'failed'
  completed_at: string | null,       // ISO timestamp for completed/failed
  error_message: string | null,      // Error details if failed
  metadata: {                        // Enhanced metadata object
    file_path?: string,              // Resolved file path
    enhanced_prompt?: string,        // AI-enhanced prompt
    callback_processed_at: string,   // Processing timestamp
    callback_debug: object,          // Debug information
    model_type: 'sdxl' | 'wan',     // Model identification
    bucket: string,                  // Storage bucket name
    is_sdxl: boolean,               // SDXL flag
    file_path_validation: object,   // Path validation details
    debug_info: object              // Additional debugging
  }
}

// Images Table Updates (Image Jobs)
{
  status: 'completed' | 'failed' | 'generating',
  image_url: string,                 // Primary image URL
  image_urls: string[] | null,       // Multiple URLs (SDXL batch)
  thumbnail_url: string,             // Thumbnail reference
  quality: 'fast' | 'high',          // Generation quality
  metadata: {                        // Enhanced metadata
    model_type: 'sdxl' | 'wan',
    is_sdxl: boolean,
    bucket: string,
    callback_processed_at: string,
    file_path_validation: object,
    debug_info: object
  }
}

// Videos Table Updates (Video Jobs)
{
  status: 'completed' | 'failed' | 'processing',
  video_url: string,                 // Video file URL
  completed_at: string,              // Completion timestamp
  error_message?: string             // Error details if failed
}
```

**File Path Handling**:
```typescript
// Path Normalization Function
function normalizeAssetPath(filePath, userId) {
  if (!filePath || !userId) return filePath;
  
  // Check if path already contains user ID prefix
  if (filePath.startsWith(`${userId}/`)) {
    return filePath; // Already user-scoped
  }
  
  // Add user ID prefix for consistency
  return `${userId}/${filePath}`;
}

// SDXL Path Handling
// Images: User-scoped paths with SDXL prefix
// Pattern: `${userId}/sdxl_${jobId}_*.png`
// Batch: Multiple URLs in `assets` array
// Primary: First image in array becomes `image_url`

// WAN Path Handling
// Images: User-scoped paths without prefix
// Videos: Bucket root paths (just filename)
// Pattern: `${userId}/${jobId}_*.png` or `${jobId}_*.mp4`
```

**Error Handling & Validation**:
```typescript
// Critical Validations
const jobId = requestBody.job_id || requestBody.jobId;
if (!jobId) {
  throw new Error('job_id or jobId is required');
}

const resolvedFilePath = requestBody.assets?.[0] || requestBody.filePath || requestBody.outputUrl;
if (!resolvedFilePath && status === 'completed') {
  console.error('❌ CRITICAL: No file path provided for completed job');
}

// File Path Validation
const filePathValidation = {
  hasSlash: primaryImageUrl ? primaryImageUrl.includes('/') : false,
  hasUnderscore: primaryImageUrl ? primaryImageUrl.includes('_') : false,
  hasPngExtension: primaryImageUrl ? primaryImageUrl.endsWith('.png') : false,
  length: primaryImageUrl ? primaryImageUrl.length : 0,
  startsWithUserId: primaryImageUrl ? primaryImageUrl.startsWith(job.user_id) : false,
  expectedPattern: `${job.user_id}/${isSDXL ? 'sdxl_' : ''}${job.id}_*.png`,
  isMultipleImages: !!imageUrlsArray,
  imageCount: imageUrlsArray ? imageUrlsArray.length : 1
};
```

---

## **Parameter Consistency Standards**

### **Worker Callback Format**
All workers now use **standardized callback parameters**:

```typescript
// CONSISTENT across SDXL, WAN, and orchestrator:
{
  job_id: string,           // Database job ID
  status: 'completed' | 'failed' | 'processing',  // Job status
  assets: string[],         // Array of file URLs
  error_message?: string    // Error details (optional)
}
```

### **Job Payload Format**
All edge functions create **standardized job payloads**:

```typescript
// CONSISTENT job payload structure:
{
  id: string,              // Database job ID
  type: string,            // Job type
  prompt: string,          // User prompt
  user_id: string,         // User identifier
  config: object,          // Generation configuration
  created_at: string,      // ISO timestamp
  video_id?: string,       // Optional video ID
  image_id?: string,       // Optional image ID
  metadata?: object        // Additional metadata
}
```

### **File Path Standards**
**SDXL Jobs**:
- Pattern: `${userId}/sdxl_${jobId}_${timestamp}_${index}.png`
- Batch: Array of 6 URLs
- Storage: `sdxl_image_fast` or `sdxl_image_high` bucket

**WAN Jobs**:
- Images: `${userId}/${jobId}_${timestamp}.png`
- Videos: `${jobId}_${timestamp}.mp4`
- Storage: Job-type specific bucket

### **Job Type Mapping**
```yaml
SDXL Jobs:
  sdxl_image_fast: Queue: sdxl_queue, Bucket: sdxl_image_fast
  sdxl_image_high: Queue: sdxl_queue, Bucket: sdxl_image_high

Standard WAN Jobs:
  image_fast: Queue: wan_queue, Bucket: image_fast
  image_high: Queue: wan_queue, Bucket: image_high
  video_fast: Queue: wan_queue, Bucket: video_fast
  video_high: Queue: wan_queue, Bucket: video_high

Enhanced WAN Jobs:
  image7b_fast_enhanced: Queue: wan_queue, Bucket: image7b_fast_enhanced
  image7b_high_enhanced: Queue: wan_queue, Bucket: image7b_high_enhanced
  video7b_fast_enhanced: Queue: wan_queue, Bucket: video7b_fast_enhanced
  video7b_high_enhanced: Queue: wan_queue, Bucket: video7b_high_enhanced
```

**Status: 🚧 TESTING PHASE - 5/10 Job Types Verified** 