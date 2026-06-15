import { useState, useEffect, useCallback } from 'react'
import ImageSelector from './components/ImageSelector.jsx'
import AnnotationCanvas from './components/AnnotationCanvas.jsx'
import BboxList from './components/BboxList.jsx'
import { fetchImages, fetchImageUrl, fetchAnnotation, saveAnnotation, inferBbox } from './api.js'

let bboxCounter = 0
const newId = () => `bbox-${++bboxCounter}`

export default function App() {
  const [images, setImages] = useState([])
  const [selected, setSelected] = useState('')
  const [bboxes, setBboxes] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [inferring, setInferring] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')

  useEffect(() => {
    fetchImages().then(setImages).catch(console.error)
  }, [])

  const handleImageChange = useCallback(async (filename) => {
    setSelected(filename)
    setBboxes([])
    setSelectedId(null)
    if (!filename) return
    const data = await fetchAnnotation(filename)
    setBboxes(data.connectors.map(c => ({ ...c, id: c.id || newId() })))
  }, [])

  const handleBboxAdd = useCallback((coords) => {
    const id = newId()
    setBboxes(prev => [...prev, { id, ...coords, category: '', vlm_text: null, vlm_shape: null }])
    setSelectedId(id)
  }, [])

  const handleCategoryChange = useCallback((id, value) => {
    setBboxes(prev => prev.map(b => b.id === id ? { ...b, category: value } : b))
  }, [])

  const handleDelete = useCallback((id) => {
    setBboxes(prev => prev.filter(b => b.id !== id))
    setSelectedId(prev => prev === id ? null : prev)
  }, [])

  const handleClearAll = useCallback(() => {
    setBboxes([])
    setSelectedId(null)
  }, [])

  const handleInfer = useCallback(async (bbox) => {
    setInferring(bbox.id)
    try {
      const result = await inferBbox(selected, {
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
  }, [selected])

  const handleSave = useCallback(async () => {
    if (!selected) return
    setSaving(true)
    setSaveStatus('')
    try {
      await saveAnnotation(selected, { image: selected, connectors: bboxes })
      setSaveStatus('保存完了')
    } catch (e) {
      setSaveStatus('保存失敗')
      console.error(e)
    } finally {
      setSaving(false)
      setTimeout(() => setSaveStatus(''), 2000)
    }
  }, [selected, bboxes])

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px', borderBottom: '1px solid #1e3a5f', background: '#112236' }}>
        <ImageSelector images={images} selected={selected} onChange={handleImageChange} />
        <div style={{ display: 'flex', gap: '8px', padding: '6px 0' }}>
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
            disabled={!selected || saving}
            style={{
              padding: '6px 16px',
              background: saveStatus === '保存完了' ? '#1b5e20' : saveStatus === '保存失敗' ? '#b71c1c' : !selected || saving ? '#2a3f55' : '#1a6b8a',
              color: '#fff', border: 'none', borderRadius: '4px',
              cursor: !selected || saving ? 'not-allowed' : 'pointer',
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
        <div style={{ flex: 1, overflow: 'auto', padding: '8px', display: 'flex', alignItems: 'flex-start' }}>
          <AnnotationCanvas
            imageSrc={selected ? fetchImageUrl(selected) : ''}
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
            onInfer={handleInfer}
            onDelete={handleDelete}
            inferring={inferring}
          />
        </div>
      </div>
    </div>
  )
}
