import { useState, useRef, useCallback, useEffect } from 'react'
import GalleryPage from './components/GalleryPage.jsx'
import AnnotationCanvas from './components/AnnotationCanvas.jsx'
import BboxList from './components/BboxList.jsx'
import { fetchImageUrl, fetchAnnotation, saveAnnotation, inferBbox } from './api.js'

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
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const savedSnapshot = useRef('')
  const bboxesRef = useRef([])
  const bboxHistory = useRef([]) // undo stack

  // Keep bboxesRef in sync so handlers can read latest value without deps
  useEffect(() => { bboxesRef.current = bboxes }, [bboxes])

  const isDirty = snapshot(bboxes, completed) !== savedSnapshot.current

  const pushHistory = () => {
    bboxHistory.current = [...bboxHistory.current.slice(-49), bboxesRef.current]
  }

  const handleUndo = useCallback(() => {
    if (bboxHistory.current.length === 0) return
    const prev = bboxHistory.current[bboxHistory.current.length - 1]
    bboxHistory.current = bboxHistory.current.slice(0, -1)
    setBboxes(prev)
    setSelectedId(null)
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
    bboxHistory.current = []
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
  }, [])

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

  if (page === 'gallery') {
    return <GalleryPage key={galleryKey} onSelectImage={openEditor} />
  }

  const canUndo = bboxHistory.current.length > 0

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
              await saveAnnotation(selected, { image: selected, connectors: bboxes, completed })
              setPage('gallery')
              setGalleryKey(k => k + 1)
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
            disabled={!canUndo}
            title="元に戻す (Ctrl+Z)"
            style={{
              padding: '6px 12px', borderRadius: '4px', fontSize: '14px',
              border: `1px solid ${canUndo ? '#4a6a8a' : '#2a3f55'}`,
              background: canUndo ? '#1e3448' : '#161e2a',
              color: canUndo ? '#c0d8f0' : '#3a5070',
              cursor: canUndo ? 'pointer' : 'not-allowed',
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
        <div style={{ width: '260px', borderLeft: '1px solid #1e3a5f', display: 'flex', flexDirection: 'column', background: '#0f2035' }}>
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
