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
  const [inferring, setInferring] = useState(null)
  const [saveStatus, setSaveStatus] = useState('')

  useEffect(() => {
    fetchImages().then(setImages).catch(console.error)
  }, [])

  const handleImageChange = useCallback(async (filename) => {
    setSelected(filename)
    setBboxes([])
    if (!filename) return
    const data = await fetchAnnotation(filename)
    setBboxes(data.connectors.map(c => ({ ...c, id: c.id || newId() })))
  }, [])

  const handleBboxAdd = useCallback((coords) => {
    setBboxes(prev => [...prev, { id: newId(), ...coords, category: '', vlm_text: null, vlm_shape: null }])
  }, [])

  const handleCategoryChange = useCallback((id, value) => {
    setBboxes(prev => prev.map(b => b.id === id ? { ...b, category: value } : b))
  }, [])

  const handleDelete = useCallback((id) => {
    setBboxes(prev => prev.filter(b => b.id !== id))
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
    setSaveStatus('保存中...')
    try {
      await saveAnnotation(selected, { image: selected, connectors: bboxes })
      setSaveStatus('保存完了')
    } catch (e) {
      setSaveStatus('保存失敗')
      console.error(e)
    }
    setTimeout(() => setSaveStatus(''), 2000)
  }, [selected, bboxes])

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px' }}>
        <ImageSelector images={images} selected={selected} onChange={handleImageChange} />
        <button
          onClick={handleSave}
          disabled={!selected}
          style={{
            padding: '6px 16px', background: selected ? '#2e7d32' : '#555',
            color: '#fff', border: 'none', borderRadius: '4px', cursor: selected ? 'pointer' : 'not-allowed',
          }}
        >
          {saveStatus || '保存'}
        </button>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflow: 'auto', padding: '8px', display: 'flex', alignItems: 'flex-start' }}>
          <AnnotationCanvas
            imageSrc={selected ? fetchImageUrl(selected) : ''}
            bboxes={bboxes}
            onBboxAdd={handleBboxAdd}
          />
        </div>
        <div style={{ width: '260px', borderLeft: '1px solid #333', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '8px', borderBottom: '1px solid #333', fontSize: '13px', color: '#aaa' }}>
            Bbox 一覧 ({bboxes.length})
          </div>
          <BboxList
            bboxes={bboxes}
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
