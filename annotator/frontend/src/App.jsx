import { useState, useRef, useCallback, useEffect } from 'react'
import { useBboxHistory } from './hooks/useBboxHistory.js'
import GalleryPage from './components/GalleryPage.jsx'
import MaskingGalleryPage from './components/MaskingGalleryPage.jsx'
import MaskingEditor from './components/MaskingEditor.jsx'
import ResultsGalleryPage from './components/ResultsGalleryPage.jsx'
import ResultsViewer from './components/ResultsViewer.jsx'
import AnnotationCanvas from './components/AnnotationCanvas.jsx'
import BboxList from './components/BboxList.jsx'
import { fetchImageUrl, fetchAnnotation, saveAnnotation, inferBbox } from './api.js'

function NavBar({ currentPage, onNavigate }) {
  const items = [
    { key: 'gallery', label: 'アノテーション', color: '#5aabff', border: '#1e3a5f' },
    { key: 'masking_gallery', label: 'マスキング', color: '#c4aaff', border: '#3a2a6e' },
    { key: 'results_gallery', label: '結果表示', color: '#a8e6a8', border: '#2a5a2e' },
  ]
  return (
    <div style={{
      display: 'flex', gap: '2px', padding: '6px 12px',
      background: '#0a1420', borderBottom: '1px solid #1a2a3a', flexShrink: 0,
    }}>
      {items.map(({ key, label, color, border }) => {
        const active = currentPage === key
        return (
          <button
            key={key}
            onClick={() => onNavigate(key)}
            style={{
              padding: '6px 18px', borderRadius: '4px', fontSize: '13px',
              border: `1px solid ${active ? color : border}`,
              background: active ? 'rgba(255,255,255,0.07)' : 'transparent',
              color: active ? color : '#7a9cc0',
              cursor: 'pointer', fontWeight: active ? 'bold' : 'normal',
              transition: 'all 0.15s',
            }}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

const newId = () => crypto.randomUUID()

const snapshot = (bboxes, completed) => JSON.stringify({ bboxes, completed })

export default function App() {
  const [page, setPage] = useState('gallery')
  const [galleryKey, setGalleryKey] = useState(0)
  const [selected, setSelected] = useState('')
  const [imageDir, setImageDir] = useState('samples')
  const [bboxes, setBboxes] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [completed, setCompleted] = useState(false)
  const [inferring, setInferring] = useState(null)
  const [galleryDir, setGalleryDir] = useState('')
  const [maskingSelected, setMaskingSelected] = useState('')
  const [maskingImageDir, setMaskingImageDir] = useState('samples')
  const [maskingGalleryDir, setMaskingGalleryDir] = useState('')
  const [resultsImage, setResultsImage] = useState('')
  const [resultsDir, setResultsDir] = useState('samples')
  const [resultsGalleryDir, setResultsGalleryDir] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const [sidebarWidth, setSidebarWidth] = useState(260)
  const [dividerHover, setDividerHover] = useState(false)
  const savedSnapshot = useRef('')
  const isDragging = useRef(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)
  const { bboxesRef, syncRef, push: pushHistory, undo: handleUndo, reset: resetHistory, canUndo } = useBboxHistory(setBboxes, setSelectedId)

  // Keep bboxesRef in sync so handlers can read latest value without deps
  useEffect(() => { syncRef(bboxes) }, [bboxes, syncRef])

  const isDirty = snapshot(bboxes, completed) !== savedSnapshot.current

  // Divider drag
  const handleDividerMouseDown = useCallback((e) => {
    isDragging.current = true
    dragStartX.current = e.clientX
    dragStartWidth.current = sidebarWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [sidebarWidth])

  useEffect(() => {
    const onMove = (e) => {
      if (!isDragging.current) return
      const delta = dragStartX.current - e.clientX
      setSidebarWidth(Math.max(160, Math.min(600, dragStartWidth.current + delta)))
    }
    const onUp = () => {
      if (!isDragging.current) return
      isDragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  // Ctrl+Z undo (editor only)
  useEffect(() => {
    if (page !== 'editor') return
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        handleUndo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [page, handleUndo])

  const openEditor = useCallback(async (filename, dir = 'samples') => {
    resetHistory()
    setSelected(filename)
    setImageDir(dir)
    setBboxes([])
    setSelectedId(null)
    setCompleted(false)
    const data = await fetchAnnotation(filename)
    const loadedBboxes = data.connectors.map(c => ({ part_number: '', notes: '', ...c, id: c.id || newId() }))
    const loadedCompleted = data.completed ?? false
    setBboxes(loadedBboxes)
    setCompleted(loadedCompleted)
    savedSnapshot.current = snapshot(loadedBboxes, loadedCompleted)
    setPage('editor')
  }, [resetHistory])

  const handleBboxAdd = useCallback((coords) => {
    pushHistory()
    const id = newId()
    setBboxes(prev => [...prev, { id, ...coords, category: '', part_number: '', notes: '', vlm_text: null, vlm_shape: null }])
    setSelectedId(id)
  }, [])

  const handleCategoryChange = useCallback((id, value) => {
    setBboxes(prev => prev.map(b => b.id === id ? { ...b, category: value } : b))
  }, [])

  const handleFieldChange = useCallback((id, field, value) => {
    setBboxes(prev => prev.map(b => b.id === id ? { ...b, [field]: value } : b))
  }, [])

  const handleDelete = useCallback((id) => {
    pushHistory()
    setBboxes(prev => prev.filter(b => b.id !== id))
    setSelectedId(prev => prev === id ? null : prev)
  }, [])

  const handleClearAll = useCallback(() => {
    pushHistory()
    setBboxes([])
    setSelectedId(null)
  }, [])

  const handleInfer = useCallback(async (bbox) => {
    setInferring(bbox.id)
    try {
      const result = await inferBbox(selected, imageDir, {
        x_center: bbox.x_center,
        y_center: bbox.y_center,
        width: bbox.width,
        height: bbox.height,
      })
      setBboxes(prev => prev.map(b =>
        b.id === bbox.id ? { ...b, vlm_text: result.vlm_text, vlm_shape: result.vlm_shape } : b
      ))
    } catch (e) {
      console.error('Inference failed:', e)
    } finally {
      setInferring(null)
    }
  }, [selected, imageDir])

  const handleSave = useCallback(async () => {
    if (!selected) return
    setSaving(true)
    setSaveStatus('')
    try {
      await saveAnnotation(selected, { image: selected, connectors: bboxes, completed })
      savedSnapshot.current = snapshot(bboxes, completed)
      setSaveStatus('保存完了')
    } catch (e) {
      setSaveStatus('保存失敗')
      console.error(e)
    } finally {
      setSaving(false)
      setTimeout(() => setSaveStatus(''), 2000)
    }
  }, [selected, bboxes, completed])

  const handleToggleCompleted = useCallback(async () => {
    const next = !completed
    setCompleted(next)
    await saveAnnotation(selected, { image: selected, connectors: bboxes, completed: next })
    savedSnapshot.current = snapshot(bboxes, next)
  }, [selected, bboxes, completed])

  const openMaskingEditor = useCallback((filename, dir) => {
    setMaskingSelected(filename)
    setMaskingImageDir(dir)
    setPage('masking_editor')
  }, [])

  const openResultsViewer = useCallback((filename, dir) => {
    setResultsImage(filename)
    setResultsDir(dir)
    setPage('results_viewer')
  }, [])

  if (['gallery', 'masking_gallery', 'results_gallery'].includes(page)) {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0d1b2a' }}>
        <NavBar currentPage={page} onNavigate={setPage} />
        {page === 'gallery' && (
          <GalleryPage key={galleryKey} onSelectImage={openEditor} selectedDir={galleryDir} onDirChange={setGalleryDir} />
        )}
        {page === 'masking_gallery' && (
          <MaskingGalleryPage onSelectImage={openMaskingEditor} selectedDir={maskingGalleryDir} onDirChange={setMaskingGalleryDir} />
        )}
        {page === 'results_gallery' && (
          <ResultsGalleryPage onSelectImage={openResultsViewer} selectedDir={resultsGalleryDir} onDirChange={setResultsGalleryDir} />
        )}
      </div>
    )
  }

  if (page === 'masking_editor') {
    return (
      <MaskingEditor
        filename={maskingSelected}
        imageDir={maskingImageDir}
        onBack={() => setPage('masking_gallery')}
      />
    )
  }

  if (page === 'results_viewer') {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0d1b2a', overflow: 'hidden' }}>
        <ResultsViewer
          filename={resultsImage}
          dir={resultsDir}
          onBack={() => setPage('results_gallery')}
        />
      </div>
    )
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 8px', borderBottom: '1px solid #1e3a5f', background: '#112236',
      }}>
        <span style={{ fontSize: '13px', color: '#e2eaf5', padding: '0 4px' }}>{selected}</span>
        <div style={{ display: 'flex', gap: '8px', padding: '6px 0', alignItems: 'center' }}>
          <button
            onClick={async () => {
              try {
                await saveAnnotation(selected, { image: selected, connectors: bboxes, completed })
              } catch (e) {
                console.error('Save on back failed:', e)
              } finally {
                setPage('gallery')
                setGalleryKey(k => k + 1)
              }
            }}
            style={{
              padding: '6px 14px', borderRadius: '4px', fontSize: '14px',
              border: '1px solid #3a6090', background: '#1e3a5c',
              color: '#a8ccee', cursor: 'pointer',
            }}
          >
            一覧へ
          </button>
          <button
            onClick={handleToggleCompleted}
            title={completed ? 'クリックで未完了に戻す' : 'クリックで完了にする'}
            style={{
              padding: '6px 14px', borderRadius: '4px', fontSize: '14px',
              border: `1px solid ${completed ? '#00e676' : '#4a6a8a'}`,
              background: completed ? '#0d3320' : '#1e3448',
              color: completed ? '#00e676' : '#c0d8f0',
              cursor: 'pointer', transition: 'all 0.2s',
            }}
          >
            {completed ? '✓ 完了済み' : '○ 未完了'}
          </button>
          <button
            onClick={handleUndo}
            disabled={!canUndo()}
            title="元に戻す (Ctrl+Z)"
            style={{
              padding: '6px 12px', borderRadius: '4px', fontSize: '14px',
              border: `1px solid ${canUndo() ? '#4a6a8a' : '#2a3f55'}`,
              background: canUndo() ? '#1e3448' : '#161e2a',
              color: canUndo() ? '#c0d8f0' : '#3a5070',
              cursor: canUndo() ? 'pointer' : 'not-allowed',
            }}
          >
            ↩ Undo
          </button>
          <button
            onClick={handleClearAll}
            disabled={bboxes.length === 0}
            style={{
              padding: '6px 14px',
              background: bboxes.length === 0 ? '#2a3f55' : '#7f0000',
              color: '#fff', border: 'none', borderRadius: '4px',
              cursor: bboxes.length === 0 ? 'not-allowed' : 'pointer',
              fontSize: '14px',
            }}
          >
            全消去
          </button>
          <button
            onClick={handleSave}
            disabled={!isDirty || saving}
            style={{
              padding: '6px 16px',
              background: saveStatus === '保存完了' ? '#1b5e20' : saveStatus === '保存失敗' ? '#b71c1c' : !isDirty || saving ? '#2a3f55' : '#1a6b8a',
              color: !isDirty && !saving ? '#4a6a8a' : '#fff',
              border: 'none', borderRadius: '4px',
              cursor: !isDirty || saving ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', gap: '6px',
              fontSize: '14px', minWidth: '90px', justifyContent: 'center',
              transition: 'background 0.2s',
            }}
          >
            {saving && <span className="spinner" />}
            {saving ? '保存中...' : saveStatus || '保存'}
          </button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflow: 'hidden', background: '#0d1b2a' }}>
          <AnnotationCanvas
            imageSrc={selected ? fetchImageUrl(selected, imageDir) : ''}
            bboxes={bboxes}
            selectedId={selectedId}
            onBboxAdd={handleBboxAdd}
            onSelect={setSelectedId}
          />
        </div>

        {/* Resizer */}
        <div
          onMouseDown={handleDividerMouseDown}
          onMouseEnter={() => setDividerHover(true)}
          onMouseLeave={() => setDividerHover(false)}
          style={{
            width: '5px', flexShrink: 0, cursor: 'col-resize',
            background: dividerHover ? '#2a6090' : '#1a3050',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.15s',
            position: 'relative',
          }}
        >
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '3px', pointerEvents: 'none',
          }}>
            {[0,1,2].map(i => (
              <div key={i} style={{
                width: '3px', height: '3px', borderRadius: '50%',
                background: dividerHover ? '#7ab8e8' : '#3a5a7a',
              }} />
            ))}
          </div>
        </div>

        <div style={{ width: `${sidebarWidth}px`, display: 'flex', flexDirection: 'column', background: '#0f2035', flexShrink: 0 }}>
          <div style={{ padding: '8px', borderBottom: '1px solid #1e3a5f', fontSize: '13px', color: '#7a9cc0' }}>
            Bbox 一覧 ({bboxes.length})
          </div>
          <BboxList
            bboxes={bboxes}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onCategoryChange={handleCategoryChange}
            onFieldChange={handleFieldChange}
            onInfer={handleInfer}
            onDelete={handleDelete}
            inferring={inferring}
          />
        </div>
      </div>
    </div>
  )
}
