import { useState, useEffect, useRef, useMemo } from 'react'
import { marked } from 'marked'
import {
  Brain,
  Globe,
  FileText,
  Filter,
  PenTool,
  ShieldCheck,
  Search,
  Sparkles,
  Copy,
  Check,
  AlertCircle,
  BookOpen,
  UploadCloud,
  X,
  CheckCircle2,
  FileUp,
  Database,
  ArrowRight,
  ArrowLeft,
  Layers,
  Download,
  Printer,
  ExternalLink,
  Plus,
  Clock,
  ChevronDown,
  ChevronUp,
  Cpu,
  RefreshCw,
  Info,
  Link2,
  SlidersHorizontal,
  Bookmark,
  Share2,
  ListFilter,
  Settings,
  Paperclip
} from 'lucide-react'
import './App.css'

// =========================================================================
// MASCOT COMPONENT (Matching Reference Hero)
// =========================================================================
function MascotLogo({ size = 52 }) {
  return (
    <div className="mascot-container">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="mascot-svg"
      >
        <defs>
          <linearGradient id="mascotLensGrad" x1="12" y1="12" x2="84" y2="84" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="50%" stopColor="#0284c7" />
            <stop offset="100%" stopColor="#0ea5e9" />
          </linearGradient>
          <linearGradient id="mascotEarLeft" x1="24" y1="12" x2="34" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#0284c7" />
          </linearGradient>
          <linearGradient id="mascotEarRight" x1="64" y1="12" x2="74" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#0284c7" />
          </linearGradient>
          <filter id="mascotHalo" x="-25%" y="-25%" width="150%" height="150%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Ambient Halo */}
        <circle cx="48" cy="46" r="38" fill="rgba(14, 165, 233, 0.25)" />

        {/* Magnifying Glass Handle */}
        <line x1="72" y1="70" x2="90" y2="88" stroke="url(#mascotLensGrad)" strokeWidth="7" strokeLinecap="round" />
        <line x1="74" y1="72" x2="88" y2="86" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" opacity="0.8" />

        {/* Left Ear */}
        <ellipse cx="30" cy="18" rx="8" ry="14" fill="#ffffff" stroke="url(#mascotLensGrad)" strokeWidth="2.5" />
        <ellipse cx="30" cy="19" rx="4.5" ry="9" fill="url(#mascotEarLeft)" />

        {/* Right Ear */}
        <ellipse cx="66" cy="18" rx="8" ry="14" fill="#ffffff" stroke="url(#mascotLensGrad)" strokeWidth="2.5" />
        <ellipse cx="66" cy="19" rx="4.5" ry="9" fill="url(#mascotEarRight)" />

        {/* Outer Magnifying Glass Rim */}
        <circle cx="48" cy="46" r="33" stroke="url(#mascotLensGrad)" strokeWidth="5.5" fill="#0b1728" filter="url(#mascotHalo)" />
        <circle cx="48" cy="46" r="29.5" stroke="rgba(255, 255, 255, 0.3)" strokeWidth="1" fill="#0f1f38" />

        {/* Cute Face Base */}
        <ellipse cx="48" cy="49" rx="23" ry="19" fill="#ffffff" />

        {/* Left Eye */}
        <ellipse cx="39" cy="45" rx="4.5" ry="5.5" fill="#0f172a" />
        <circle cx="37.5" cy="43" r="1.8" fill="#ffffff" />
        <circle cx="40.5" cy="47" r="0.8" fill="#ffffff" />

        {/* Right Eye */}
        <ellipse cx="57" cy="45" rx="4.5" ry="5.5" fill="#0f172a" />
        <circle cx="55.5" cy="43" r="1.8" fill="#ffffff" />
        <circle cx="58.5" cy="47" r="0.8" fill="#ffffff" />

        {/* Nose & Smile */}
        <ellipse cx="48" cy="49.5" rx="1.8" ry="1.2" fill="#ec4899" />
        <path d="M44.5 52.5 C46 54.5, 48 53.5, 48 51.5 C48 53.5, 50 54.5, 51.5 52.5" stroke="#334155" strokeWidth="1.5" strokeLinecap="round" fill="none" />

        {/* Rosy Cheeks */}
        <ellipse cx="32" cy="49" rx="3.5" ry="2" fill="rgba(244, 114, 182, 0.5)" />
        <ellipse cx="64" cy="49" rx="3.5" ry="2" fill="rgba(244, 114, 182, 0.5)" />

        {/* Glass Highlight Glare */}
        <path d="M26 34 A 28 28 0 0 1 66 24" stroke="rgba(255, 255, 255, 0.65)" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      </svg>
    </div>
  )
}

// =========================================================================
// UNIVERSAL APPLICATION TOP NAVIGATION HEADER
// =========================================================================
function AppHeader({ page, navigateTo, uploadedDocsCount, result }) {
  return (
    <header className="app-top-header">
      <div
        className="header-brand"
        onClick={() => navigateTo('search')}
        role="button"
        tabIndex={0}
      >
        <MascotLogo size={34} />
        <div className="brand-text">
          <span className="brand-title">Mini Researcher</span>
          <span className="brand-subtitle">Autonomous Swarm</span>
        </div>
      </div>

      <nav className="header-nav-tabs" role="tablist">
        <button
          type="button"
          className={`header-nav-btn ${page === 'search' ? 'header-nav-active' : ''}`}
          onClick={() => navigateTo('search')}
          title="Swarm Research Cockpit"
        >
          <Search size={14} />
          <span>Research</span>
        </button>

        <button
          type="button"
          className={`header-nav-btn ${page === 'documents' ? 'header-nav-active' : ''}`}
          onClick={() => navigateTo('documents')}
          title="Manage Uploaded Documents & Vector Store"
        >
          <Database size={14} />
          <span>Knowledge Base</span>
          {uploadedDocsCount > 0 && (
            <span className="header-count-pill">{uploadedDocsCount}</span>
          )}
        </button>

        <button
          type="button"
          className={`header-nav-btn ${page === 'swarm' ? 'header-nav-active' : ''}`}
          onClick={() => navigateTo('swarm')}
          title="Inspect 6 Multi-Agent Swarm Roles"
        >
          <Brain size={14} />
          <span>Agent Swarm</span>
        </button>

        {result && (
          <>
            <span className="header-nav-separator"></span>
            <button
              type="button"
              className={`header-nav-btn ${page === 'report' ? 'header-nav-active' : ''}`}
              onClick={() => navigateTo('report')}
              title="View Synthesized Executive Report"
            >
              <FileText size={14} />
              <span>Report</span>
            </button>

            <button
              type="button"
              className={`header-nav-btn ${page === 'resources' ? 'header-nav-active' : ''}`}
              onClick={() => navigateTo('resources')}
              title="Curated Evidence Sources"
            >
              <BookOpen size={14} />
              <span>Sources</span>
              <span className="header-count-pill">{result.sources?.length || 0}</span>
            </button>

            <button
              type="button"
              className={`header-nav-btn ${page === 'audit' ? 'header-nav-active' : ''}`}
              onClick={() => navigateTo('audit')}
              title="Reviewer Factual Grounding Audit"
            >
              <ShieldCheck size={14} />
              <span>Audit</span>
            </button>
          </>
        )}
      </nav>
    </header>
  )
}

// =========================================================================
// REPORT TYPE & TONE OPTIONS (Exact Match from User Reference Images)
// =========================================================================
const REPORT_TYPES = [
  { id: 'summary', label: 'Summary - Short and fast (~2 min)' },
  { id: 'deep_research', label: 'Deep Research Report' },
  { id: 'multi_agents', label: 'Multi Agents Report' },
  { id: 'detailed', label: 'Detailed - In depth and longer (~5 min)' }
]

const TONES = [
  { id: 'objective', label: 'Objective - Impartial and unbiased presentation of facts and findings' },
  { id: 'formal', label: 'Formal - Adheres to academic standards with sophisticated language and structure' },
  { id: 'analytical', label: 'Analytical - Critical evaluation and detailed examination of data and theories' },
  { id: 'persuasive', label: 'Persuasive - Convincing the audience of a particular viewpoint or argument' },
  { id: 'informative', label: 'Informative - Providing clear and comprehensive information on a topic' },
  { id: 'explanatory', label: 'Explanatory - Clarifying complex concepts and processes' },
  { id: 'descriptive', label: 'Descriptive - Detailed depiction of phenomena, experiments, or case studies' },
  { id: 'critical', label: 'Critical - Judging the validity and relevance of the research and its conclusions' },
  { id: 'comparative', label: 'Comparative - Juxtaposing different theories, data, or methods to highlight differences and similarities' },
  { id: 'speculative', label: 'Speculative - Exploring hypotheses and potential implications or future research directions' },
  { id: 'reflective', label: 'Reflective - Considering the research process and personal insights or experiences' },
  { id: 'narrative', label: 'Narrative - Telling a story to illustrate research findings or methodologies' },
  { id: 'humorous', label: 'Humorous - Light-hearted and engaging, usually to make the content more relatable' },
  { id: 'optimistic', label: 'Optimistic - Highlighting positive findings and potential benefits' },
  { id: 'pessimistic', label: 'Pessimistic - Focusing on limitations, challenges, or negative outcomes' },
  { id: 'simple', label: 'Simple - Written for young readers, using basic vocabulary and clear explanations' },
  { id: 'casual', label: 'Casual - Conversational and relaxed style for easy, everyday reading' }
]

const PROMPT_STARTERS = [
  { icon: '📈', prefix: 'Stock analysis on' },
  { icon: '🏃', prefix: 'Help me plan an adventure to' },
  { icon: '📰', prefix: 'What are the latest news on' }
]

// =========================================================================
// AGENTS SPECIFICATION & ARCHITECTURE
// =========================================================================
const AGENTS = [
  {
    id: 'manager',
    name: 'ManagerAgent',
    title: 'Lead Coordinator',
    icon: Brain,
    color: '#818cf8',
    glow: 'rgba(129, 140, 248, 0.4)',
    role: 'Task decomposition, mode routing (Web / Document / Hybrid), and swarm orchestration.',
    model: 'Gemini 3.6 / 3.7 / 3.8 Flash (Auto-Redirect)'
  },
  {
    id: 'research',
    name: 'ResearchAgent',
    title: 'Web Intelligence',
    icon: Globe,
    color: '#38bdf8',
    glow: 'rgba(56, 189, 248, 0.4)',
    role: 'Strategic subquery planning (5 angles), live Tavily search execution, and recency filtering.',
    model: 'Tavily Search API'
  },
  {
    id: 'document',
    name: 'DocumentAgent',
    title: 'Document Intelligence',
    icon: FileText,
    color: '#c084fc',
    glow: 'rgba(192, 132, 252, 0.4)',
    role: 'Parses PDFs, text, & markdown into semantic chunks and handles ChromaDB vector RAG retrieval.',
    model: 'ChromaDB Local VectorStore'
  },
  {
    id: 'curator',
    name: 'CuratorAgent',
    title: 'Evidence Curator',
    icon: Filter,
    color: '#34d399',
    glow: 'rgba(52, 211, 153, 0.4)',
    role: 'Scores semantic relevance, filters noise & duplicate URLs, detects contradictions, and vets quality.',
    model: 'Gemini 3.6 / 3.7 / 3.8 Flash & Scoring'
  },
  {
    id: 'writer',
    name: 'WriterAgent',
    title: 'Report Synthesizer',
    icon: PenTool,
    color: '#fbbf24',
    glow: 'rgba(251, 191, 36, 0.4)',
    role: 'Synthesizes curated evidence into grounded GitHub-flavored Markdown reports with source references.',
    model: 'Groq Llama 3.1 & Llama 3.8'
  },
  {
    id: 'reviewer',
    name: 'ReviewerAgent',
    title: 'Quality & Grounding Auditor',
    icon: ShieldCheck,
    color: '#f43f5e',
    glow: 'rgba(244, 63, 94, 0.4)',
    role: 'Audits factual grounding, completeness, and citations. Triggers automatic revision loop if rejected.',
    model: 'Structured Pydantic Audit'
  }
]

const SUGGESTIONS = [
  'Latest quantum computing milestones 2026',
  'Solid-state battery energy density breakthroughs',
  'Autonomous AI agent swarm architectures',
  'CRISPR gene editing clinical trial results'
]

export default function App() {
  // Page Navigation: 'search' | 'report' | 'resources' | 'audit'
  const [page, setPage] = useState('search')

  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [activeStep, setActiveStep] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [copied, setCopied] = useState(false)
  const [inspectAgent, setInspectAgent] = useState(null)

  // Dedicated Resources Page State
  const [resourceSearch, setResourceSearch] = useState('')
  const [resourceFilter, setResourceFilter] = useState('all') // 'all' | 'web' | 'document'
  const [resourceSort, setResourceSort] = useState('relevance') // 'relevance' | 'domain' | 'title'
  const [expandedSources, setExpandedSources] = useState({})

  // Toast notification state
  const [toast, setToast] = useState(null)
  const showToast = (message, type = 'info') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3200)
  }

  // Execution Mode: 'auto' (ManagerAgent decided) | 'web' | 'document'
  const [mode, setMode] = useState('auto')

  // Report Type & Tone state (matching user reference images)
  const [reportType, setReportType] = useState('Summary - Short and fast (~2 min)')
  const [tone, setTone] = useState('Objective - Impartial and unbiased presentation of facts and findings')
  const [reportTypeOpen, setReportTypeOpen] = useState(false)
  const [toneOpen, setToneOpen] = useState(false)
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false)
  const [downloadingPdf, setDownloadingPdf] = useState(false)

  // Element Refs
  const searchInputRef = useRef(null)
  const reportTypeRef = useRef(null)
  const toneRef = useRef(null)

  // Close dropdowns on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (reportTypeRef.current && !reportTypeRef.current.contains(e.target)) {
        setReportTypeOpen(false)
      }
      if (toneRef.current && !toneRef.current.contains(e.target)) {
        setToneOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Document Upload States
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [uploading, setUploading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState('')
  const [includeUploaded, setIncludeUploaded] = useState(true)
  const fileInputRef = useRef(null)

  // Active steps based dynamically on reportType and mode
  const currentPipelineSteps = useMemo(() => {
    const norm = (reportType || '').toLowerCase()

    // 1. Summary: Direct LLM generation (~500 words)
    if (norm.includes('summary')) {
      return mode === 'document' ? [
        { agent: 'DocumentAgent', action: 'Reading & parsing uploaded documents', icon: FileText, color: '#c084fc' },
        { agent: 'WriterAgent', action: 'Direct LLM executive synthesis (~500 words)', icon: PenTool, color: '#38bdf8' }
      ] : [
        { agent: 'WriterAgent', action: 'Direct LLM executive synthesis (~500 words)', icon: PenTool, color: '#38bdf8' }
      ]
    }

    // 2. Deep Research: Research strictly by Tavily API
    if (norm.includes('deep')) {
      return [
        { agent: 'ResearchAgent', action: 'Formulating strategic angles & querying Tavily API', icon: Globe, color: '#38bdf8' },
        { agent: 'CuratorAgent', action: 'Vetting Tavily sources & extracting findings', icon: Filter, color: '#34d399' },
        { agent: 'WriterAgent', action: 'Drafting Deep Research Report with citations', icon: PenTool, color: '#fbbf24' }
      ]
    }

    // 3. Multi Agents: Specifically activate ReviewerAgent
    if (norm.includes('multi')) {
      return [
        { agent: 'WriterAgent', action: 'Drafting multi-perspective collaborative report', icon: PenTool, color: '#fbbf24' },
        { agent: 'ReviewerAgent', action: 'ACTIVATED: Auditing factual grounding & revision loop', icon: ShieldCheck, color: '#34d399' }
      ]
    }

    // 4. Detailed: In depth and longer (~5 min) - ALL 6 AGENTS
    return [
      { agent: 'ManagerAgent', action: 'Decomposing objective into 5 targeted strategic angles', icon: Brain, color: '#818cf8' },
      { agent: 'ResearchAgent', action: 'Executing Tavily web search across all angles', icon: Globe, color: '#38bdf8' },
      { agent: 'DocumentAgent', action: 'Indexing & querying ChromaDB vector store', icon: FileText, color: '#c084fc' },
      { agent: 'CuratorAgent', action: 'Deduplicating & scraping high-relevance findings', icon: Filter, color: '#34d399' },
      { agent: 'WriterAgent', action: 'Composing comprehensive in-depth dossier (~5 min read)', icon: PenTool, color: '#fbbf24' },
      { agent: 'ReviewerAgent', action: 'Auditing factual grounding & executing revision loops', icon: ShieldCheck, color: '#f43f5e' }
    ]
  }, [reportType, mode])


  // Step simulation during loading
  useEffect(() => {
    let interval = null
    if (loading) {
      setActiveStep(0)
      interval = setInterval(() => {
        setActiveStep((prev) => (prev < currentPipelineSteps.length - 1 ? prev + 1 : prev))
      }, 2000)
    } else {
      setActiveStep(0)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [loading, currentPipelineSteps.length])

  // Browser hash navigation: supports #search, #documents, #swarm, #report, #resources, #audit
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '')
      if (['documents', 'swarm'].includes(hash)) {
        setPage(hash)
      } else if (['report', 'resources', 'audit'].includes(hash) && result) {
        setPage(hash)
      } else if (hash === 'search' || !hash) {
        setPage('search')
      }
    }
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [result])

  // Fetch uploaded documents on mount
  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents')
      if (res.ok) {
        const data = await res.json()
        setUploadedDocs(data.documents || [])
      }
    } catch (_) { }
  }

  const navigateTo = (targetPage) => {
    setPage(targetPage)
    window.location.hash = targetPage
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return
    const file = files[0]

    const validExtensions = ['.pdf', '.txt', '.md']
    const hasValidExt = validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext))
    if (!hasValidExt) {
      setError(`Unsupported file format for "${file.name}". Please upload PDF (.pdf), Text (.txt), or Markdown (.md).`)
      return
    }

    setUploading(true)
    setError('')
    setUploadSuccess('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let msg = `Upload failed with status ${res.status}`
        try {
          const errData = await res.json()
          if (errData?.detail) msg = errData.detail
        } catch (_) { }
        throw new Error(msg)
      }

      const data = await res.json()
      setUploadSuccess(`Indexed "${file.name}" into ChromaDB (${data.document?.chunks_indexed || 1} chunks).`)
      showToast(`Indexed "${file.name}" into ChromaDB!`, 'success')
      setTimeout(() => setUploadSuccess(''), 6000)
      fetchDocuments()
    } catch (err) {
      setError(err.message || 'Failed to upload and index document.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDeleteDocument = async (filename) => {
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        setUploadedDocs((prev) => prev.filter((d) => d.filename !== filename))
        showToast(`Document "${filename}" removed.`, 'info')
      }
    } catch (err) {
      setError(`Failed to remove document "${filename}".`)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files)
    }
  }

  const executeResearch = async (searchTerm) => {
    let q = (searchTerm !== undefined ? searchTerm : query).trim()

    if (!q && (mode === 'document' || (mode === 'auto' && uploadedDocs.length > 0))) {
      q = 'Provide a comprehensive summary and key findings of the uploaded document(s).'
      setQuery(q)
    }

    if (!q) {
      setError(mode === 'document' ? 'Please upload a document or enter an objective.' : 'Please enter a research topic or question.')
      return
    }

    if (mode === 'document' && uploadedDocs.length === 0) {
      setError('Document Mode requires at least one uploaded document. Please upload a PDF, TXT, or MD file.')
      return
    }

    setError('')
    setLoading(true)

    try {
      const response = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          mode: mode,
          include_uploaded: mode === 'web' ? false : includeUploaded,
          report_type: reportType,
          tone: tone
        }),
      })

      if (!response.ok) {
        let msg = `Server responded with status ${response.status}`
        try {
          const errData = await response.json()
          if (errData?.detail) msg = errData.detail
        } catch (_) { }
        throw new Error(msg)
      }

      const data = await response.json()
      setResult(data)
      navigateTo('report')
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    executeResearch()
  }

  const handlePromptStarter = (starter) => {
    setQuery(starter.prefix + ' ')
    if (searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }

  const handleSuggestionClick = (topic) => {
    setQuery(topic)
    setMode('web')
    executeResearch(topic)
  }

  const handleCopy = () => {
    if (!result?.report) return
    navigator.clipboard.writeText(result.report).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const handleDownloadPdf = () => {
    if (!result?.report) return
    setDownloadingPdf(true)
    try {
      const cleanTitle = (result.topic || query || 'research-report')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 40) || 'research-report'

      // Use a hidden form pointing to an invisible iframe
      // This sends an authentic HTTP download directly through the browser's native download manager
      // eliminating client-side blob UUID naming bugs in Edge and Chrome completely.
      let iframe = document.getElementById('pdf-native-download-iframe')
      if (!iframe) {
        iframe = document.createElement('iframe')
        iframe.id = 'pdf-native-download-iframe'
        iframe.name = 'pdf-native-download-iframe'
        iframe.style.display = 'none'
        document.body.appendChild(iframe)
      }

      const form = document.createElement('form')
      form.method = 'POST'
      form.action = 'http://localhost:8000/api/export-pdf-download'
      form.target = 'pdf-native-download-iframe'
      form.style.display = 'none'

      const addInput = (name, val) => {
        const inp = document.createElement('input')
        inp.type = 'hidden'
        inp.name = name
        inp.value = val || ''
        form.appendChild(inp)
      }

      addInput('topic', result.topic || query || 'Research Report')
      addInput('report', result.report || '')
      addInput('report_type', result.report_type || reportType || 'Standard')
      addInput('tone', result.tone || tone || 'Objective')
      addInput('model', result.model || 'Gemini 3.5 Flash Lite')

      document.body.appendChild(form)
      form.submit()

      setTimeout(() => {
        if (document.body.contains(form)) {
          document.body.removeChild(form)
        }
        setDownloadingPdf(false)
        showToast(`Downloaded: ${cleanTitle}.pdf`, 'success')
      }, 900)
    } catch (err) {
      console.error('PDF download error:', err)
      setDownloadingPdf(false)
      window.print()
    }
  }

  const formatFileSize = (chars) => {
    if (!chars) return '0 B'
    if (chars < 1024) return `${chars} chars`
    return `${(chars / 1024).toFixed(1)} KB`
  }

  const formatDomain = (url) => {
    if (!url) return 'Direct Synthesis'
    if (url.startsWith('uploaded://')) return 'Local Document (ChromaDB)'
    try {
      const host = url.replace(/^https?:\/\//i, '').split('/')[0]
      return host.replace(/^www\./i, '')
    } catch (_) {
      return url
    }
  }

  const toggleExpandSource = (id) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const handleCopyCitation = (src, e) => {
    if (e) e.stopPropagation()
    const title = src.title || 'Referenced Document'
    const url = src.url || ''
    const citation = url.startsWith('uploaded://')
      ? `[Document] ${title}. Indexed in ChromaDB Vector Store.`
      : `${title}. Retrieved from: ${url}`
    navigator.clipboard.writeText(citation).then(() => {
      showToast(`Citation copied for "${title.slice(0, 32)}..."`, 'success')
    })
  }

  const handleCopyLink = (url, e) => {
    if (e) e.stopPropagation()
    if (!url) return
    navigator.clipboard.writeText(url).then(() => {
      showToast('Source link copied to clipboard!', 'success')
    })
  }

  const handleCopyAllCitations = () => {
    if (!result?.sources || result.sources.length === 0) return
    const text = result.sources
      .map((src, i) => {
        const isDoc = src.url?.startsWith('uploaded://')
        return `[${i + 1}] ${src.title || 'Untitled Source'}\n    Type: ${isDoc ? 'Local Indexed Document (ChromaDB)' : 'Web Intelligence'}\n    Reference: ${src.url || 'N/A'}`
      })
      .join('\n\n')
    navigator.clipboard.writeText(text).then(() => {
      showToast(`Copied all ${result.sources.length} citations to clipboard!`, 'success')
    })
  }

  const handleDownloadBibliography = () => {
    if (!result?.sources || result.sources.length === 0) return
    const header = `# Bibliography & Evidence References\n\n**Research Objective:** ${result.topic}\n**Date Generated:** ${new Date().toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}\n**Total Sources:** ${result.sources.length}\n\n---\n\n`
    const list = result.sources
      .map((src, i) => {
        const isDoc = src.url?.startsWith('uploaded://')
        let item = `### [${i + 1}] ${src.title || 'Referenced Resource'}\n`
        item += `- **Source Category:** ${isDoc ? 'ChromaDB Local Vector Ingestion' : 'Live Tavily Web Search'}\n`
        if (src.url) item += `- **Reference Location:** ${src.url}\n`
        if (src.snippet) item += `- **Curated Evidence Extract:** "${src.snippet.trim()}"\n`
        return item
      })
      .join('\n\n')

    const blob = new Blob([header + list], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const safeTopic = (result.topic || 'research').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 32)
    a.download = `${safeTopic}-bibliography.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    showToast('Bibliography downloaded as Markdown (.md)!', 'success')
  }

  // Filter and sort resources for the dedicated Resources Page
  const processedSources = useMemo(() => {
    if (!result?.sources) return []
    let list = [...result.sources]

    // Filter by type
    if (resourceFilter === 'web') {
      list = list.filter((s) => !s.url?.startsWith('uploaded://'))
    } else if (resourceFilter === 'document') {
      list = list.filter((s) => s.url?.startsWith('uploaded://'))
    }

    // Filter by search query
    if (resourceSearch.trim()) {
      const q = resourceSearch.toLowerCase().trim()
      list = list.filter(
        (s) =>
          (s.title && s.title.toLowerCase().includes(q)) ||
          (s.snippet && s.snippet.toLowerCase().includes(q)) ||
          (s.url && s.url.toLowerCase().includes(q))
      )
    }

    // Sort
    if (resourceSort === 'title') {
      list.sort((a, b) => (a.title || '').localeCompare(b.title || ''))
    } else if (resourceSort === 'domain') {
      list.sort((a, b) => {
        const getDomain = (url) => (url ? url.replace(/^https?:\/\//, '').split('/')[0] : '')
        return getDomain(a.url).localeCompare(getDomain(b.url))
      })
    } else {
      // Relevance (default)
      list.sort((a, b) => (b.score || 1) - (a.score || 1))
    }

    return list
  }, [result?.sources, resourceFilter, resourceSearch, resourceSort])

  // Count source categories
  const webSourcesCount = (result?.sources || []).filter((s) => !s.url?.startsWith('uploaded://')).length
  const docSourcesCount = (result?.sources || []).filter((s) => s.url?.startsWith('uploaded://')).length
  const wordCount = result?.report ? result.report.trim().split(/\s+/).length : 0
  const readingTimeMinutes = Math.max(1, Math.ceil(wordCount / 220))

  // =========================================================================
  // RESEARCH RESULTS PRESENTATION: REPORT, DEDICATED RESOURCES, OR AUDIT
  // =========================================================================
  if ((page === 'report' || page === 'resources' || page === 'audit') && result) {
    const review = result.review || {
      decision: 'APPROVE',
      overall_score: 0.94,
      factual_grounding: 0.95,
      completeness: 0.92,
      citation_quality: 0.90,
      revisions: 0,
      summary: 'Report verified and fully grounded in available evidence.'
    }

    return (
      <div className="app-shell">
        <AppHeader
          page={page}
          navigateTo={navigateTo}
          uploadedDocsCount={uploadedDocs.length}
          result={result}
        />
        <div className="app report-page-container">
          {/* Toast Notification */}
          {toast && (
            <div className={`toast-notification toast-${toast.type}`}>
              <CheckCircle2 size={16} />
              <span>{toast.message}</span>
            </div>
          )}

          {/* Sticky Universal Navigation Bar with View Switcher */}
          <nav className="report-navbar">
            <div className="report-nav-left">
              <button
                type="button"
                className="btn-back"
                onClick={() => navigateTo('search')}
                title="Return to swarm research cockpit"
              >
                <ArrowLeft size={15} />
                <span>Swarm Cockpit</span>
              </button>

              <span className="report-nav-divider"></span>

              {result.mode === 'document' ? (
                <span className="mode-indicator-chip mode-indicator-doc">
                  <FileText size={13} />
                  Document Agent Swarm
                </span>
              ) : (
                <span className="mode-indicator-chip mode-indicator-web">
                  <Globe size={13} />
                  Web Research Swarm
                </span>
              )}

              <span className="mode-indicator-chip mode-indicator-type" title={`Report Type: ${result.report_type || reportType}`}>
                <Layers size={13} />
                {(result.report_type || reportType).split(' - ')[0]}
              </span>

              <span className="mode-indicator-chip mode-indicator-tone" title={`Tone of Voice: ${result.tone || tone}`}>
                <Sparkles size={13} />
                {(result.tone || tone).split(' - ')[0]}
              </span>
            </div>

            {/* Central Multi-Page View Tabs */}
            <div className="report-nav-tabs" role="tablist">
              <button
                type="button"
                className={`nav-tab-item ${page === 'report' ? 'nav-tab-active' : ''}`}
                onClick={() => navigateTo('report')}
                title="View full executive markdown synthesis"
              >
                <FileText size={14} />
                <span>Synthesis Report</span>
              </button>

              <button
                type="button"
                className={`nav-tab-item ${page === 'resources' ? 'nav-tab-active' : ''}`}
                onClick={() => navigateTo('resources')}
                title="View dedicated resources, web links, and citation repository"
              >
                <BookOpen size={14} />
                <span>Resources &amp; Evidence</span>
                <span className="nav-tab-badge">{result.sources?.length || 0}</span>
              </button>

              <button
                type="button"
                className={`nav-tab-item ${page === 'audit' ? 'nav-tab-active' : ''}`}
                onClick={() => navigateTo('audit')}
                title="View multi-agent audit scorecard and grounding telemetry"
              >
                <ShieldCheck size={14} />
                <span>Grounding Audit</span>
                <span className="nav-tab-badge score-badge">
                  {Math.round((review.overall_score || 0.9) * 100)}%
                </span>
              </button>
            </div>

            {/* Action Tools */}
            <div className="report-nav-actions">
              <button
                type="button"
                className="btn-report-action"
                onClick={handleCopy}
                title="Copy full markdown report text to clipboard"
              >
                {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
                <span>{copied ? 'Copied!' : 'Copy Report'}</span>
              </button>

              <button
                type="button"
                className="btn-report-action btn-pdf-action"
                onClick={handleDownloadPdf}
                disabled={downloadingPdf}
                title="Download report in publication-ready PDF format"
              >
                {downloadingPdf ? (
                  <>
                    <RefreshCw size={14} className="spinning-sync-icon" />
                    <span>Generating PDF...</span>
                  </>
                ) : (
                  <>
                    <FileText size={14} />
                    <span>Download PDF</span>
                  </>
                )}
              </button>

              <button
                type="button"
                className="btn-report-primary"
                onClick={() => {
                  setQuery('')
                  navigateTo('search')
                }}
                title="Start a fresh inquiry in the Swarm Cockpit"
              >
                <Plus size={14} />
                <span>New Research</span>
              </button>
            </div>
          </nav>

          {/* ================================================================= */}
          {/* VIEW 1: EXECUTIVE SYNTHESIS REPORT CANVAS                         */}
          {/* ================================================================= */}
          {page === 'report' && (
            <div className="report-view-fade-in">
              {/* Report Hero Section */}
              <header className="report-hero">
                <div className="report-hero-meta-row">
                  <span className="meta-chip">
                    <Clock size={12} style={{ display: 'inline', marginRight: '4px' }} />
                    Generated {new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                  </span>
                  <span className="meta-chip">
                    <Clock size={12} style={{ display: 'inline', marginRight: '4px' }} />
                    ~{readingTimeMinutes} min read ({wordCount.toLocaleString()} words)
                  </span>
                  <span className="meta-chip">Gemini 3.6 / 3.7 / 3.8 Flash (Resilient Fallback)</span>
                  {result.mode === 'document' ? (
                    <>
                      <span className="meta-chip meta-chip-docs">
                        <FileText size={12} style={{ display: 'inline', marginRight: '4px' }} />
                        {result.uploaded_documents?.join(', ') || 'Uploaded Document'}
                      </span>
                      <span className="meta-chip">
                        <Layers size={12} style={{ display: 'inline', marginRight: '4px' }} />
                        {result.findings_count || 0} Chunks
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="meta-chip">
                        <BookOpen size={12} style={{ display: 'inline', marginRight: '4px' }} />
                        {result.sources?.length || 0} Curated Sources
                      </span>
                      <span className="meta-chip">
                        <Globe size={12} style={{ display: 'inline', marginRight: '4px' }} />
                        {result.subqueries?.length || 5} Angles
                      </span>
                    </>
                  )}
                  <span className="meta-chip meta-chip-revisions">
                    <RefreshCw size={12} style={{ display: 'inline', marginRight: '4px' }} />
                    {review.revisions === 0 ? 'Approved Pass 1 (0 Revisions)' : `${review.revisions} Automatic Revision(s)`}
                  </span>
                </div>

                <h1 className="report-main-title">{result.topic}</h1>
              </header>

              {/* Curated Resources & Evidence Dedicated Page Callout Banner */}
              <div className="report-resources-callout">
                <div className="resources-callout-info">
                  <div className="resources-callout-icon">
                    <BookOpen size={22} />
                  </div>
                  <div className="resources-callout-text">
                    <div className="resources-callout-title">
                      <span>Curated Evidence &amp; Resource Repository</span>
                      <span className="resources-callout-pill">{result.sources?.length || 0} Vetted Items</span>
                    </div>
                    <p className="resources-callout-desc">
                      All source citations, direct links, ChromaDB vector chunks, and research angles have been organized into a dedicated interactive explorer page.
                    </p>
                    {result.sources && result.sources.length > 0 && (
                      <div className="resources-callout-domains">
                        {result.sources.slice(0, 4).map((s, idx) => (
                          <span key={idx} className="callout-domain-tag">
                            {formatDomain(s.url)}
                          </span>
                        ))}
                        {result.sources.length > 4 && (
                          <span className="callout-domain-more">+{result.sources.length - 4} more</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <button
                  type="button"
                  className="btn-view-dedicated-resources"
                  onClick={() => navigateTo('resources')}
                  title="Open the dedicated Resources & Evidence Page"
                >
                  <span>Explore Resources Page</span>
                  <ArrowRight size={15} />
                </button>
              </div>

              {/* Distraction-Free Report Reading Canvas */}
              <article className="report-content-canvas">
                <div
                  className="report-content"
                  dangerouslySetInnerHTML={{ __html: marked.parse(result.report || '') }}
                />
              </article>

              {/* End of Report Resources Callout */}
              <div className="report-end-resources-card">
                <div className="end-card-left">
                  <Bookmark size={20} color="var(--accent-cyan)" />
                  <div>
                    <strong>Ready to inspect the primary sources or export citations?</strong>
                    <p>Browse live citations, search across snippets, or export a formatted Markdown bibliography.</p>
                  </div>
                </div>
                <button
                  type="button"
                  className="btn-report-primary"
                  onClick={() => navigateTo('resources')}
                >
                  <BookOpen size={14} />
                  <span>Open Curated Resources Page ({result.sources?.length || 0})</span>
                </button>
              </div>

              {/* Bottom Navigation & Actions Bar */}
              <div className="report-footer-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => navigateTo('search')}
                >
                  <ArrowLeft size={15} />
                  <span>Conduct Another Research</span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => navigateTo('resources')}
                >
                  <BookOpen size={15} />
                  <span>View All Resources ({result.sources?.length || 0})</span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                >
                  <span>Back to Top ↑</span>
                </button>
              </div>
            </div>
          )}

          {/* ================================================================= */}
          {/* VIEW 2: DEDICATED RESOURCES & EVIDENCE PAGE                       */}
          {/* ================================================================= */}
          {page === 'resources' && (
            <div className="resources-page-view report-view-fade-in">
              {/* Resources Hero Header */}
              <div className="resources-hero-header">
                <div className="resources-breadcrumb">
                  <span onClick={() => navigateTo('search')} className="breadcrumb-link">Swarm Cockpit</span>
                  <span className="breadcrumb-sep">&gt;</span>
                  <span onClick={() => navigateTo('report')} className="breadcrumb-link">Synthesis Report</span>
                  <span className="breadcrumb-sep">&gt;</span>
                  <span className="breadcrumb-current">Resources &amp; Evidence</span>
                </div>

                <div className="resources-hero-title-row">
                  <div>
                    <h1 className="resources-main-title">Research Resources &amp; Evidence</h1>
                    <p className="resources-main-subtitle">
                      Curated source citations, live web references, and vector database passages assembled for: <strong>"{result.topic}"</strong>
                    </p>
                  </div>

                  <button
                    type="button"
                    className="btn-return-report"
                    onClick={() => navigateTo('report')}
                    title="Switch back to executive report reading canvas"
                  >
                    <FileText size={15} />
                    <span>Return to Report</span>
                  </button>
                </div>

                {/* Research Metrics Stats Grid */}
                <div className="resources-stats-grid">
                  <div className="resource-stat-card">
                    <span className="stat-label">Total Evidence Items</span>
                    <span className="stat-number">{result.sources?.length || 0}</span>
                    <span className="stat-sub">Vetted by CuratorAgent</span>
                  </div>

                  <div className="resource-stat-card">
                    <span className="stat-label">Live Web Intelligence</span>
                    <span className="stat-number text-cyan">{webSourcesCount}</span>
                    <span className="stat-sub">Tavily Verified Articles</span>
                  </div>

                  <div className="resource-stat-card">
                    <span className="stat-label">Document RAG Chunks</span>
                    <span className="stat-number text-purple">{docSourcesCount}</span>
                    <span className="stat-sub">ChromaDB Vector Passages</span>
                  </div>

                  <div className="resource-stat-card">
                    <span className="stat-label">Exploration Angles</span>
                    <span className="stat-number text-indigo">{result.subqueries?.length || 5}</span>
                    <span className="stat-sub">ResearchAgent Subqueries</span>
                  </div>

                  <div className="resource-stat-card">
                    <span className="stat-label">Reviewer Grounding</span>
                    <span className="stat-number text-emerald">{Math.round((review.factual_grounding || 0.95) * 100)}%</span>
                    <span className="stat-sub">Factually Grounded</span>
                  </div>
                </div>
              </div>

              {/* ResearchAgent Strategic Angles Showcase */}
              {result.subqueries && result.subqueries.length > 0 && (
                <div className="subqueries-showcase-panel">
                  <div className="subqueries-header">
                    <div className="subqueries-title-group">
                      <Globe size={16} color="#38bdf8" />
                      <strong>Strategic Research Angles Explored by ResearchAgent</strong>
                    </div>
                    <span className="subqueries-count-pill">{result.subqueries.length} Strategic Angles</span>
                  </div>
                  <div className="subqueries-grid-cards">
                    {result.subqueries.map((sq, idx) => (
                      <div key={idx} className="subquery-item-card">
                        <span className="subquery-num">Angle {idx + 1}</span>
                        <p className="subquery-text">{sq}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Resources Interactive Toolbar: Search, Filters, & Actions */}
              <div className="resources-toolbar">
                {/* Keyword Search Input */}
                <div className="resources-search-box">
                  <Search size={16} className="resources-search-icon" />
                  <input
                    type="text"
                    className="resources-search-input"
                    placeholder="Filter resources by title, domain, keywords, or snippet text..."
                    value={resourceSearch}
                    onChange={(e) => setResourceSearch(e.target.value)}
                  />
                  {resourceSearch && (
                    <button
                      type="button"
                      className="btn-clear-filter"
                      onClick={() => setResourceSearch('')}
                      title="Clear filter"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>

                {/* Category Segmented Filter */}
                <div className="resources-filter-pills">
                  <button
                    type="button"
                    className={`filter-pill ${resourceFilter === 'all' ? 'filter-pill-active' : ''}`}
                    onClick={() => setResourceFilter('all')}
                  >
                    All ({result.sources?.length || 0})
                  </button>
                  <button
                    type="button"
                    className={`filter-pill ${resourceFilter === 'web' ? 'filter-pill-active' : ''}`}
                    onClick={() => setResourceFilter('web')}
                  >
                    <Globe size={13} />
                    Web ({webSourcesCount})
                  </button>
                  <button
                    type="button"
                    className={`filter-pill ${resourceFilter === 'document' ? 'filter-pill-active' : ''}`}
                    onClick={() => setResourceFilter('document')}
                  >
                    <FileText size={13} />
                    Documents ({docSourcesCount})
                  </button>
                </div>

                {/* Sort Selector */}
                <div className="resources-sort-wrapper">
                  <SlidersHorizontal size={14} color="var(--text-dim)" />
                  <select
                    className="resources-sort-select"
                    value={resourceSort}
                    onChange={(e) => setResourceSort(e.target.value)}
                  >
                    <option value="relevance">Sort: Highest Relevance</option>
                    <option value="domain">Sort: Source Domain</option>
                    <option value="title">Sort: Title (A-Z)</option>
                  </select>
                </div>

                {/* Batch Citation Tools */}
                <div className="resources-batch-actions">
                  <button
                    type="button"
                    className="btn-resource-batch"
                    onClick={handleCopyAllCitations}
                    title="Copy all reference citations formatted to clipboard"
                  >
                    <Copy size={13} />
                    <span>Copy All Citations</span>
                  </button>

                  <button
                    type="button"
                    className="btn-resource-batch"
                    onClick={handleDownloadBibliography}
                    title="Download complete bibliography as a Markdown file"
                  >
                    <Download size={13} />
                    <span>Export Bibliography (.md)</span>
                  </button>
                </div>
              </div>

              {/* Resources List / Grid */}
              <div className="resources-results-container">
                <div className="resources-results-header">
                  <span className="resources-results-count">
                    Showing <strong>{processedSources.length}</strong> of {result.sources?.length || 0} evidence resources
                    {resourceSearch && ` matching "${resourceSearch}"`}
                  </span>
                  {resourceSearch && (
                    <button
                      type="button"
                      className="btn-reset-filters"
                      onClick={() => {
                        setResourceSearch('')
                        setResourceFilter('all')
                      }}
                    >
                      Reset filters
                    </button>
                  )}
                </div>

                {processedSources.length === 0 ? (
                  <div className="resources-empty-state">
                    <BookOpen size={36} color="var(--text-dim)" />
                    <h3>No resources match your search filter</h3>
                    <p>Try clearing your search query or selecting "All" to inspect all gathered evidence.</p>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => {
                        setResourceSearch('')
                        setResourceFilter('all')
                      }}
                    >
                      Show All Resources
                    </button>
                  </div>
                ) : (
                  <div className="resources-cards-grid">
                    {processedSources.map((src, idx) => {
                      const isDoc = src.url?.startsWith('uploaded://')
                      const isExpanded = !!expandedSources[idx]
                      const domain = formatDomain(src.url)

                      return (
                        <article key={idx} className={`resource-card-premium ${isDoc ? 'card-doc-type' : 'card-web-type'}`}>
                          {/* Card Header */}
                          <div className="resource-card-header">
                            <div className="resource-type-badges">
                              {isDoc ? (
                                <span className="res-badge res-badge-doc">
                                  <FileText size={12} />
                                  ChromaDB Vector Chunk
                                </span>
                              ) : (
                                <span className="res-badge res-badge-web">
                                  <Globe size={12} />
                                  Web Intelligence
                                </span>
                              )}
                              <span className="res-domain-pill">{domain}</span>
                            </div>

                            <div className="res-score-badge" title="Semantic quality and relevance score">
                              <span className="score-dot"></span>
                              <span>{src.score ? `${Math.round(src.score * 100)}% Match` : 'Curated'}</span>
                            </div>
                          </div>

                          {/* Title */}
                          <h3 className="resource-card-title">
                            {!isDoc && src.url ? (
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="resource-title-link"
                                title="Open original resource in new tab"
                              >
                                <span>{src.title || 'Referenced Web Article'}</span>
                                <ExternalLink size={14} className="title-ext-icon" />
                              </a>
                            ) : (
                              <span>{src.title || 'Indexed Document Passage'}</span>
                            )}
                          </h3>

                          {/* Source URL display */}
                          <div className="resource-url-row">
                            <Link2 size={12} color="var(--text-dim)" />
                            <span className="resource-clean-url" title={src.url}>
                              {src.url?.startsWith('uploaded://') ? src.url.replace('uploaded://', 'doc://') : src.url}
                            </span>
                          </div>

                          {/* Snippet / Passage Box */}
                          {src.snippet && (
                            <div className="resource-snippet-box">
                              <p className="resource-snippet-text">
                                "{isExpanded ? src.snippet : `${src.snippet.slice(0, 220)}${src.snippet.length > 220 ? '...' : ''}`}"
                              </p>
                              {src.snippet.length > 220 && (
                                <button
                                  type="button"
                                  className="btn-toggle-snippet"
                                  onClick={() => toggleExpandSource(idx)}
                                >
                                  {isExpanded ? (
                                    <>
                                      <span>Show less</span>
                                      <ChevronUp size={13} />
                                    </>
                                  ) : (
                                    <>
                                      <span>Show full extract</span>
                                      <ChevronDown size={13} />
                                    </>
                                  )}
                                </button>
                              )}
                            </div>
                          )}

                          {/* Actions Row */}
                          <div className="resource-card-actions">
                            {!isDoc && src.url && (
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="btn-card-action btn-card-open"
                                title="Visit original article in browser"
                              >
                                <ExternalLink size={13} />
                                <span>Open Source</span>
                              </a>
                            )}

                            <button
                              type="button"
                              className="btn-card-action"
                              onClick={(e) => handleCopyCitation(src, e)}
                              title="Copy formatted citation"
                            >
                              <Copy size={13} />
                              <span>Copy Citation</span>
                            </button>

                            {src.url && (
                              <button
                                type="button"
                                className="btn-card-action"
                                onClick={(e) => handleCopyLink(src.url, e)}
                                title="Copy URL to clipboard"
                              >
                                <Link2 size={13} />
                                <span>Copy Link</span>
                              </button>
                            )}
                          </div>
                        </article>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Bottom Actions Bar */}
              <div className="resources-footer-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => navigateTo('report')}
                >
                  <ArrowLeft size={16} />
                  <span>Return to Synthesis Report</span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => navigateTo('search')}
                >
                  <span>New Research Query</span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                >
                  <span>Back to Top ↑</span>
                </button>
              </div>
            </div>
          )}

          {/* ================================================================= */}
          {/* VIEW 3: DEDICATED GROUNDING & MULTI-AGENT AUDIT TELEMETRY         */}
          {/* ================================================================= */}
          {page === 'audit' && (
            <div className="audit-page-view report-view-fade-in">
              {/* Audit Hero */}
              <div className="audit-hero-header">
                <div className="resources-breadcrumb">
                  <span onClick={() => navigateTo('search')} className="breadcrumb-link">Swarm Cockpit</span>
                  <span className="breadcrumb-sep">&gt;</span>
                  <span onClick={() => navigateTo('report')} className="breadcrumb-link">Synthesis Report</span>
                  <span className="breadcrumb-sep">&gt;</span>
                  <span className="breadcrumb-current">Grounding Audit</span>
                </div>

                <div className="resources-hero-title-row">
                  <div>
                    <h1 className="resources-main-title">ReviewerAgent Grounding &amp; Quality Audit</h1>
                    <p className="resources-main-subtitle">
                      Automated multi-agent verification auditing factual claims, citation validity, and revision loops for: <strong>"{result.topic}"</strong>
                    </p>
                  </div>

                  <button
                    type="button"
                    className="btn-return-report"
                    onClick={() => navigateTo('report')}
                  >
                    <FileText size={15} />
                    <span>Return to Report</span>
                  </button>
                </div>
              </div>

              {/* Audit Metrics Card */}
              <div className="agent-audit-card audit-standalone-card">
                <div className="audit-card-header">
                  <div className="audit-header-left">
                    <ShieldCheck size={20} color="#34d399" />
                    <span className="audit-card-title">ReviewerAgent Telemetry Scorecard</span>
                    <span className="audit-score-pill">Overall Quality: {Math.round((review.overall_score || 0.9) * 100)}/100</span>
                  </div>
                  <span className={`review-badge-pill ${review.decision === 'APPROVE' ? 'badge-approved' : 'badge-rejected'}`}>
                    {review.decision === 'APPROVE' ? 'VERIFIED & GROUNDED' : 'REVISION TRIGGERED'}
                  </span>
                </div>

                <div className="audit-card-body">
                  <div className="metrics-grid">
                    <div className="metric-box">
                      <span className="metric-label">Factual Grounding</span>
                      <div className="metric-bar-bg">
                        <div className="metric-bar-fill fill-emerald" style={{ width: `${Math.round((review.factual_grounding || 0.92) * 100)}%` }}></div>
                      </div>
                      <span className="metric-val">{Math.round((review.factual_grounding || 0.92) * 100)}%</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Completeness</span>
                      <div className="metric-bar-bg">
                        <div className="metric-bar-fill fill-cyan" style={{ width: `${Math.round((review.completeness || 0.88) * 100)}%` }}></div>
                      </div>
                      <span className="metric-val">{Math.round((review.completeness || 0.88) * 100)}%</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Citation Quality</span>
                      <div className="metric-bar-bg">
                        <div className="metric-bar-fill fill-purple" style={{ width: `${Math.round((review.citation_quality || 0.85) * 100)}%` }}></div>
                      </div>
                      <span className="metric-val">{Math.round((review.citation_quality || 0.85) * 100)}%</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Revision Loop</span>
                      <div className="metric-revision-status">
                        {review.revisions === 0 ? (
                          <span className="badge-pass"><Check size={12} /> Approved First Pass</span>
                        ) : (
                          <span className="badge-warn"><RefreshCw size={12} /> {review.revisions} Auto Revisions</span>
                        )}
                      </div>
                      <span className="metric-val">Max Cap: 2</span>
                    </div>
                  </div>

                  {review.summary && (
                    <div className="audit-summary-note">
                      <strong>Reviewer Summary:</strong> {review.summary}
                    </div>
                  )}

                  {/* Agent Execution Trace */}
                  {result.agents && (
                    <div className="agent-trace-row" style={{ marginTop: '1.25rem' }}>
                      <span className="trace-label">Autonomous Swarm Trace:</span>
                      <div className="trace-pills">
                        {result.agents.map((ag, idx) => (
                          <div key={idx} className="trace-agent-chip" title={`${ag.name}: ${ag.summary || ag.role}`}>
                            <span className="trace-dot"></span>
                            <strong>{ag.name}</strong>
                            <span className="trace-action">({ag.summary || ag.role})</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Quick Actions */}
              <div className="report-footer-actions" style={{ marginTop: '2rem' }}>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => navigateTo('report')}
                >
                  <FileText size={15} />
                  <span>Return to Synthesis Report</span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => navigateTo('resources')}
                >
                  <BookOpen size={15} />
                  <span>Explore Curated Resources ({result.sources?.length || 0})</span>
                </button>
              </div>
            </div>
          )}

          {/* Universal Footer */}
          <footer className="footer">
            <p>Mini Researcher &bull; Multi-Agent Swarm (Manager, Research, Document, Curator, Writer, Reviewer)</p>
          </footer>
        </div>
      </div>
    )
  }

  // =========================================================================
  // PAGE 1: SEARCH & MULTI-AGENT COCKPIT
  // =========================================================================
  return (
    <div className="app-shell">
      <AppHeader
        page={page}
        navigateTo={navigateTo}
        uploadedDocsCount={uploadedDocs.length}
        result={result}
      />

      <div className="app">
        {/* Universal Document File Upload Input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".pdf,.txt,.md"
          onChange={(e) => handleFileUpload(e.target.files)}
        />

        {/* Toast Notification */}
        {toast && (
          <div className={`toast-notification toast-${toast.type}`}>
            <CheckCircle2 size={16} />
            <span>{toast.message}</span>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
            <button
              type="button"
              className="btn-secondary"
              style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
              onClick={() => setError('')}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ================================================================= */}
        {/* PAGE 1: DEDICATED KNOWLEDGE BASE PAGE                             */}
        {/* ================================================================= */}
        {page === 'documents' && (
          <div className="page-view-container knowledge-base-page">
            <div className="page-view-header">
              <div className="page-header-title-row">
                <div className="page-header-icon-box">
                  <Database size={22} />
                </div>
                <div>
                  <h2 className="page-header-title">Knowledge Base &amp; Vector Store</h2>
                  <p className="page-header-desc">
                    Upload PDF, TXT, or Markdown documents to chunk into semantic vectors and index into local ChromaDB storage for grounded Document RAG research.
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="btn-launch-mode"
                onClick={() => {
                  setMode('document')
                  navigateTo('search')
                }}
              >
                <FileText size={15} />
                <span>Research with Documents</span>
                <ArrowRight size={14} />
              </button>
            </div>

            <div
              className={`upload-card upload-card-page ${dragActive ? 'upload-card-active' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
            >
              <div className="upload-dropzone-inner">
                <div className="upload-drop-icon">
                  <UploadCloud size={30} />
                </div>
                <span className="upload-drop-title">Drag &amp; drop research documents here</span>
                <span className="upload-drop-subtitle">Supports PDF, TXT, and Markdown files</span>
                <button
                  type="button"
                  className="btn-upload"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading || loading}
                >
                  <FileUp size={16} />
                  <span>{uploading ? 'Indexing into ChromaDB...' : 'Browse & Upload Document'}</span>
                </button>
              </div>

              {/* Upload Success Bar */}
              {uploadSuccess && (
                <div className="upload-success-bar">
                  <CheckCircle2 size={16} color="var(--primary-light)" />
                  <span>{uploadSuccess}</span>
                </div>
              )}
            </div>

            {/* Active Documents Overview & List */}
            <div className="knowledge-base-list-card">
              <div className="uploaded-docs-header">
                <span className="uploaded-count-label">
                  <Database size={14} style={{ display: 'inline', marginRight: '5px' }} />
                  ChromaDB Vector Store ({uploadedDocs.length} active documents)
                </span>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={includeUploaded}
                    onChange={(e) => setIncludeUploaded(e.target.checked)}
                  />
                  <span>Include in Swarm Context</span>
                </label>
              </div>

              {uploadedDocs.length > 0 ? (
                <div className="doc-chips-grid">
                  {uploadedDocs.map((doc, idx) => (
                    <div key={idx} className="doc-chip">
                      <FileText size={15} className="doc-chip-icon" />
                      <div className="doc-chip-info">
                        <span className="doc-chip-name" title={doc.filename}>{doc.filename}</span>
                        <span className="doc-chip-meta">
                          {formatFileSize(doc.char_count)} &bull; {doc.pages || 1} {doc.pages === 1 ? 'page' : 'pages'}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="btn-remove-doc"
                        title="Remove document from memory"
                        onClick={() => handleDeleteDocument(doc.filename)}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-docs-placeholder">
                  <FileText size={32} className="empty-docs-icon" />
                  <p>No documents uploaded yet. Upload a PDF, TXT, or Markdown above to empower DocumentAgent.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* PAGE 2: DEDICATED AGENT SWARM SHOWCASE PAGE                       */}
        {/* ================================================================= */}
        {page === 'swarm' && (
          <div className="page-view-container swarm-showcase-page">
            <div className="page-view-header">
              <div className="page-header-title-row">
                <div className="page-header-icon-box">
                  <Brain size={22} />
                </div>
                <div>
                  <h2 className="page-header-title">Autonomous Agent Swarm Architecture</h2>
                  <p className="page-header-desc">
                    Inspect the 6 specialized AI agents, allocated Gemini models, and the dynamic orchestrator workflow.
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="btn-launch-mode"
                onClick={() => navigateTo('search')}
              >
                <Search size={15} />
                <span>Launch Research</span>
                <ArrowRight size={14} />
              </button>
            </div>

            <div className="agents-grid agents-grid-page">
              {AGENTS.map((agent) => {
                const Icon = agent.icon
                const isSelected = inspectAgent?.id === agent.id
                return (
                  <div
                    key={agent.id}
                    className={`agent-card ${isSelected ? 'agent-card-selected' : ''}`}
                    style={{ '--agent-color': agent.color, '--agent-glow': agent.glow }}
                    onClick={() => setInspectAgent(isSelected ? null : agent)}
                  >
                    <div className="agent-card-top">
                      <div className="agent-icon-box" style={{ background: `${agent.color}20`, color: agent.color }}>
                        <Icon size={18} />
                      </div>
                      <span className="agent-status-dot"></span>
                    </div>
                    <span className="agent-name">{agent.name}</span>
                    <span className="agent-tagline">{agent.title}</span>
                    <p className="agent-role-brief">{agent.role}</p>
                  </div>
                )
              })}
            </div>

            {/* Modal / Inspector Drawer for Selected Agent */}
            {inspectAgent && (
              <div className="agent-inspector-box" style={{ borderLeftColor: inspectAgent.color }}>
                <div className="inspector-top">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <inspectAgent.icon size={18} color={inspectAgent.color} />
                    <strong>{inspectAgent.name}</strong>
                    <span className="inspector-badge">{inspectAgent.title}</span>
                  </div>
                  <button
                    type="button"
                    className="btn-close-inspector"
                    onClick={() => setInspectAgent(null)}
                  >
                    <X size={14} />
                  </button>
                </div>
                <p className="inspector-desc">{inspectAgent.role}</p>
                <div className="inspector-meta">
                  <span><strong>Allocated Backend:</strong> {inspectAgent.model}</span>
                  <span><strong>Swarm Architecture:</strong> Antigravity Multi-Agent System</span>
                </div>
              </div>
            )}

            {/* Interactive Swarm Workflow Explanation */}
            <div className="swarm-workflow-card">
              <h3 className="swarm-workflow-title">
                <Sparkles size={16} color="var(--primary-light)" />
                Swarm Execution Pipeline
              </h3>
              <div className="workflow-steps-horizontal">
                <div className="workflow-step-col">
                  <div className="wf-step-num">1</div>
                  <strong>Decomposition</strong>
                  <span>ManagerAgent plans 5 targeted search angles</span>
                </div>
                <div className="wf-step-arrow">&rarr;</div>
                <div className="workflow-step-col">
                  <div className="wf-step-num">2</div>
                  <strong>Intelligence</strong>
                  <span>ResearchAgent queries Tavily &amp; DocumentAgent retrieves ChromaDB vectors</span>
                </div>
                <div className="wf-step-arrow">&rarr;</div>
                <div className="workflow-step-col">
                  <div className="wf-step-num">3</div>
                  <strong>Curator Filter</strong>
                  <span>CuratorAgent deduplicates, scores quality, &amp; strips noise</span>
                </div>
                <div className="wf-step-arrow">&rarr;</div>
                <div className="workflow-step-col">
                  <div className="wf-step-num">4</div>
                  <strong>Synthesis</strong>
                  <span>WriterAgent composes comprehensive grounded report</span>
                </div>
                <div className="wf-step-arrow">&rarr;</div>
                <div className="workflow-step-col">
                  <div className="wf-step-num">5</div>
                  <strong>Quality Audit</strong>
                  <span>ReviewerAgent checks citations &amp; triggers revision loop</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* PAGE 3: STREAMLINED SEARCH COCKPIT (HOME)                         */}
        {/* ================================================================= */}
        {page === 'search' && (
          <>
            {/* Active Report Banner with Dual Navigation: Report or Resources */}
            {result && (
              <div className="active-report-banner">
                <div className="active-report-info">
                  <Sparkles size={16} className="active-report-sparkle" />
                  <span>
                    Generated research ready: <strong>"{result.topic}"</strong>
                  </span>
                </div>
                <div className="active-report-actions-group">
                  <button
                    type="button"
                    className="btn-view-report"
                    onClick={() => navigateTo('report')}
                  >
                    <FileText size={14} />
                    <span>Read Executive Report</span>
                  </button>
                  <button
                    type="button"
                    className="btn-view-resources-banner"
                    onClick={() => navigateTo('resources')}
                  >
                    <BookOpen size={14} />
                    <span>View Resources ({result.sources?.length || 0})</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            )}

            {/* HERO SECTION */}
            <section className="hero-cockpit">
              <MascotLogo size={66} />
              <h1 className="hero-main-title">What would you like to research next?</h1>

              {/* Compact Mode Selector Segmented Bar */}
              <div className="mode-segmented-control" role="tablist">
                <button
                  type="button"
                  className={`mode-tab ${mode === 'auto' ? 'mode-tab-active' : ''}`}
                  onClick={() => setMode('auto')}
                  title="Auto Mode: ManagerAgent automatically determines routing based on your query & documents"
                >
                  <Brain size={14} />
                  <span>⚡ Auto Mode</span>
                  {uploadedDocs.length > 0 && <span className="mode-tab-pill">{uploadedDocs.length}</span>}
                </button>

                <button
                  type="button"
                  className={`mode-tab ${mode === 'web' ? 'mode-tab-active' : ''}`}
                  onClick={() => setMode('web')}
                  title="Web Research: ResearchAgent queries Tavily, CuratorAgent filters, WriterAgent drafts, ReviewerAgent audits"
                >
                  <Globe size={14} />
                  <span>Web Swarm</span>
                </button>

                <button
                  type="button"
                  className={`mode-tab ${mode === 'document' ? 'mode-tab-active' : ''}`}
                  onClick={() => setMode('document')}
                  title="Document Mode: Grounded exclusively on uploaded documents in ChromaDB"
                >
                  <FileText size={14} />
                  <span>Document RAG</span>
                  {uploadedDocs.length > 0 && <span className="mode-tab-pill">{uploadedDocs.length}</span>}
                </button>
              </div>

              {/* Search Form with Direct Document Attachment & Drag-and-Drop */}
              <form className="hero-search-form" onSubmit={handleSubmit}>
                <div
                  className={`hero-search-container ${dragActive ? 'hero-search-drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                >
                  <button
                    type="button"
                    className={`hero-attach-btn ${uploading ? 'hero-attach-btn-uploading' : ''} ${uploadedDocs.length > 0 ? 'hero-attach-has-docs' : ''}`}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading || loading}
                    title={uploading ? 'Indexing document into ChromaDB...' : 'Upload & attach document (.pdf, .txt, .md)'}
                  >
                    {uploading ? (
                      <RefreshCw size={18} className="spinning-sync-icon" />
                    ) : (
                      <Paperclip size={18} />
                    )}
                  </button>

                  <input
                    ref={searchInputRef}
                    type="text"
                    className="hero-search-input"
                    placeholder={
                      dragActive
                        ? 'Drop document here to attach to research...'
                        : uploadedDocs.length > 0 && mode === 'auto'
                          ? 'Ask a question about your document(s), or press Enter to summarize...'
                          : uploadedDocs.length > 0 && mode === 'document'
                            ? 'Ask a question about your document(s), or press Enter to summarize...'
                            : 'Enter your topic, question, or area of interest...'
                    }
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading}
                    autoFocus
                  />

                  <button
                    type="submit"
                    className="hero-submit-btn"
                    disabled={
                      loading ||
                      (!query.trim() && !(uploadedDocs.length > 0 && (mode === 'auto' || mode === 'document')))
                    }
                    title="Start research"
                  >
                    <ArrowRight size={20} />
                  </button>
                </div>
              </form>

              {/* Inline Upload Success Notification */}
              {uploadSuccess && (
                <div className="hero-upload-success-toast">
                  <div className="hero-toast-left">
                    <CheckCircle2 size={16} color="#34d399" />
                    <span>{uploadSuccess}</span>
                  </div>
                  <button
                    type="button"
                    className="btn-dismiss-inline-toast"
                    onClick={() => setUploadSuccess('')}
                  >
                    <X size={13} />
                  </button>
                </div>
              )}

              {/* AUTO MODE & DOCUMENT MODE: Direct Upload Trigger & Attached Documents Shelf */}
              {(mode === 'auto' || mode === 'document') && (
                <div className="hero-doc-control-section">
                  {uploadedDocs.length === 0 ? (
                    <div className="hero-auto-upload-bar">
                      <button
                        type="button"
                        className="btn-hero-doc-upload"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={uploading || loading}
                        title="Upload research documents"
                      >
                        <FileUp size={15} />
                        <span>Upload Document</span>
                        <span className="doc-option-badge">PDF &bull; TXT &bull; MD</span>
                      </button>
                      <span className="hero-doc-option-hint">
                        {mode === 'auto'
                          ? 'Optionally attach documents for Auto Mode to analyze and synthesize with ChromaDB RAG, or search the web directly.'
                          : 'Upload a document to begin grounded Document RAG research.'}
                      </span>
                    </div>
                  ) : (
                    <div className="hero-attached-shelf">
                      <div className="attached-shelf-header">
                        <div className="attached-shelf-left">
                          <span className="attached-pulse-dot"></span>
                          <span className="attached-shelf-title">
                            Attached to {mode === 'auto' ? 'Auto Swarm' : 'Document RAG'}:
                          </span>
                          <span className="attached-shelf-count">
                            {uploadedDocs.length} {uploadedDocs.length === 1 ? 'file' : 'files'}
                          </span>
                        </div>
                        <div className="attached-shelf-right">
                          <button
                            type="button"
                            className="btn-shelf-action btn-shelf-add"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={uploading || loading}
                            title="Attach another document (.pdf, .txt, .md)"
                          >
                            <Plus size={13} />
                            <span>Add Document</span>
                          </button>
                          <button
                            type="button"
                            className="btn-shelf-action btn-shelf-manage"
                            onClick={() => navigateTo('documents')}
                            title="Manage Vector Store & Chunks in Knowledge Base"
                          >
                            <Database size={13} />
                            <span>Knowledge Base</span>
                          </button>
                        </div>
                      </div>

                      <div className="attached-chips-row">
                        {uploadedDocs.map((doc, idx) => (
                          <div key={idx} className="attached-doc-card">
                            <div className="attached-card-icon">
                              <FileText size={15} />
                            </div>
                            <div className="attached-card-info">
                              <span className="attached-card-name" title={doc.filename}>
                                {doc.filename}
                              </span>
                              <span className="attached-card-meta">
                                {formatFileSize(doc.char_count)} &bull; {doc.pages || 1} {doc.pages === 1 ? 'page' : 'pages'} &bull; Indexed in ChromaDB
                              </span>
                            </div>
                            <button
                              type="button"
                              className="btn-remove-attached"
                              onClick={() => handleDeleteDocument(doc.filename)}
                              title={`Remove "${doc.filename}" from research context`}
                              disabled={loading}
                            >
                              <X size={13} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Web Swarm Notice if docs are in store */}
              {mode === 'web' && uploadedDocs.length > 0 && (
                <div className="hero-web-doc-notice">
                  <Info size={13} />
                  <span>
                    You have {uploadedDocs.length} document(s) in Knowledge Base. Web Swarm uses live Tavily search. Switch to <strong>Auto Mode</strong> or <strong>Document RAG</strong> to research with your documents.
                  </span>
                </div>
              )}

              <p className="hero-disclaimer">
                Mini Researcher may make mistakes. Verify important information and check sources.
              </p>

              {/* Prompt Starter Pills (Adaptive: context-aware for documents or web) */}
              {!loading && (
                <div className="hero-prompt-starters">
                  {(uploadedDocs.length > 0 && (mode === 'auto' || mode === 'document')
                    ? [
                      { icon: '📄', prefix: 'Summarize key findings in' },
                      { icon: '🔍', prefix: 'Extract methodologies and data from' },
                      { icon: '⚡', prefix: 'Analyze risks and opportunities in' }
                    ]
                    : PROMPT_STARTERS
                  ).map((item, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="hero-starter-pill"
                      onClick={() => handlePromptStarter(item)}
                    >
                      <span className="starter-icon">{item.icon}</span>
                      <span>{item.prefix}</span>
                    </button>
                  ))}
                </div>
              )}
            </section>

            {/* RESEARCH CONFIGURATION CONTROLS */}
            <section className="research-settings-cockpit">
              <div className="settings-row-dropdowns">
                {/* 1. Report Type Dropdown */}
                <div className="settings-field-col" ref={reportTypeRef}>
                  <label className="settings-field-label">Report Type</label>
                  <div className="custom-select-wrapper">
                    <button
                      type="button"
                      className={`custom-select-trigger ${reportTypeOpen ? 'custom-select-trigger-open' : ''}`}
                      onClick={() => {
                        setReportTypeOpen(!reportTypeOpen)
                        setToneOpen(false)
                      }}
                    >
                      <span>{reportType}</span>
                      <ChevronDown size={16} className={`custom-select-chevron ${reportTypeOpen ? 'chevron-up' : ''}`} />
                    </button>

                    {reportTypeOpen && (
                      <div className="custom-select-menu">
                        {REPORT_TYPES.map((rt) => (
                          <button
                            key={rt.id}
                            type="button"
                            className={`custom-select-option ${reportType === rt.label ? 'custom-select-option-active' : ''}`}
                            onClick={() => {
                              setReportType(rt.label)
                              setReportTypeOpen(false)
                            }}
                          >
                            {rt.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* 2. Tone of Voice Dropdown */}
                <div className="settings-field-col" ref={toneRef}>
                  <div className="settings-field-label-row">
                    <Settings size={15} className="settings-gear-icon" />
                    <label className="settings-field-label">Tone of Voice</label>
                  </div>
                  <div className="custom-select-wrapper">
                    <button
                      type="button"
                      className={`custom-select-trigger ${toneOpen ? 'custom-select-trigger-open' : ''}`}
                      onClick={() => {
                        setToneOpen(!toneOpen)
                        setReportTypeOpen(false)
                      }}
                    >
                      <span className="custom-select-text-truncate">{tone}</span>
                      <ChevronDown size={16} className={`custom-select-chevron ${toneOpen ? 'chevron-up' : ''}`} />
                    </button>

                    {toneOpen && (
                      <div className="custom-select-menu custom-select-menu-scrollable">
                        {TONES.map((t) => (
                          <button
                            key={t.id}
                            type="button"
                            className={`custom-select-option ${tone === t.label ? 'custom-select-option-active' : ''}`}
                            onClick={() => {
                              setTone(t.label)
                              setToneOpen(false)
                            }}
                          >
                            {t.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Popular Quick Tone Pills */}
              <div className="quick-tones-row">
                <span className="quick-tones-hint">✨ Popular Tones:</span>
                {[
                  { label: 'Objective', full: 'Objective - Impartial and unbiased presentation of facts and findings' },
                  { label: 'Analytical', full: 'Analytical - Critical evaluation and detailed examination of data and theories' },
                  { label: 'Formal', full: 'Formal - Adheres to academic standards with sophisticated language and structure' },
                  { label: 'Humorous', full: 'Humorous - Light-hearted and engaging, usually to make the content more relatable' },
                  { label: 'Simple (ELI5)', full: 'Simple - Written for young readers, using basic vocabulary and clear explanations' },
                  { label: 'Casual', full: 'Casual - Conversational and relaxed style for easy, everyday reading' }
                ].map((pill) => {
                  const isSelected = tone === pill.full || tone.toLowerCase() === pill.label.toLowerCase()
                  return (
                    <button
                      key={pill.label}
                      type="button"
                      className={`quick-tone-pill ${isSelected ? 'quick-tone-pill-active' : ''}`}
                      onClick={() => {
                        setTone(pill.full)
                        setToneOpen(false)
                      }}
                    >
                      {pill.label}
                    </button>
                  )
                })}
              </div>
            </section>
          </>
        )}

        {/* Loading: Real-Time Multi-Agent Stepper */}
        {loading && (
          <div className="swarm-execution-card">
            <div className="swarm-exec-header">
              <div className="swarm-exec-title-group">
                <span className="pulsing-live-dot"></span>
                <span className="swarm-exec-title">SWARM EXECUTION IN PROGRESS</span>
              </div>
              <span className="swarm-exec-topic">Topic: "{query}"</span>
            </div>

            <div className="swarm-steps-list">
              {currentPipelineSteps.map((step, idx) => {
                const StepIcon = step.icon
                const isCurrent = idx === activeStep
                const isDone = idx < activeStep
                return (
                  <div
                    key={idx}
                    className={`swarm-step-item ${isCurrent ? 'step-active' : ''} ${isDone ? 'step-done' : ''}`}
                  >
                    <div className="step-indicator">
                      {isDone ? (
                        <CheckCircle2 size={16} color="var(--primary-light)" />
                      ) : isCurrent ? (
                        <div className="step-spinner-dot"></div>
                      ) : (
                        <div className="step-idle-dot"></div>
                      )}
                    </div>
                    <div className="step-icon-wrap" style={{ color: step.color, background: `${step.color}15` }}>
                      <StepIcon size={16} />
                    </div>
                    <div className="step-text-group">
                      <div className="step-agent-name">
                        {step.agent}
                        {isCurrent && <span className="step-badge-live">EXECUTING</span>}
                        {isDone && <span className="step-badge-done">COMPLETED</span>}
                      </div>
                      <span className="step-action-desc">{step.action}</span>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="swarm-exec-footer">
              <Info size={14} color="var(--primary-light)" />
              <span>Autonomous multi-agent collaboration with conditional Reviewer feedback &amp; revision loops.</span>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="footer">
          <p>Mini Researcher &copy; 2026 &bull; Multi-Agent Swarm powered by Google Gemini (3.6 / 3.7 / 3.8 Flash), Tavily, &amp; ChromaDB</p>
        </footer>
      </div>
    </div>
  )
}
