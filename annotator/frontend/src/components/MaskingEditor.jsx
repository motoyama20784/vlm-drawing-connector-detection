import { useState, useEffect, useRef, useCallback } from 'react'
import AnnotationCanvas from './AnnotationCanvas.jsx'
import { fetchImageUrl, fetchMaskingFonts, fetchMaskingStatus, fetchMaskingBboxes, applyMasking } from '../api.js'

// 両キャンバスで zoom を共有するための hook
function useSharedZoom() {
  const [sharedZoom, setSharedZoom] = useState(null)
  return { sharedZoom, setSharedZoom }
}

const BBOX_COLOR = '#7c4dff'

export default function MaskingEditor({ filename, imageDir, onBack }) {
  const [bboxes, setBboxes] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [fonts, setFonts] = useState([])
  const [fontName, setFontName] = useState('')
  const [applying, setApplying] = useState(false)
  const [applyStatus, setApplyStatus] = useState('') // '' | 'success' | 'error'
  const [applyResult, setApplyResult] = useState(null)
  const [maskedImageSrc, setMaskedImageSrc] = useState(null)
  const { sharedZoom, setSharedZoom } = useSharedZoom()
  const bboxHistory = useRef([])
  const bboxesRef = useRef([])

  useEffect(() => { bboxesRef.current = bboxes }, [bboxes])

  useEffect(() => {
    fetchMaskingFonts().then(f => {
      setFonts(f)
      if (f.length > 0) setFontName(f[0].name)
    }).catch(console.error)
  }, [])

  // Load saved bboxes from previous session
  useEffect(() => {
    fetchMaskingBboxes(filename, imageDir).then(savedBboxes => {
      if (savedBboxes.length > 0) {
        setBboxes(savedBboxes.map(b => ({ id: crypto.randomUUID(), ...b })))
      }
    }).catch(() => {})
  }, [filename, imageDir])

  // Load existing masked image if it already exists from a previous session
  useEffect(() => {
    fetchMaskingStatus(imageDir).then(images => {
      const entry = images.find(img => img.filename === filename)
      if (entry?.masked) {
        setMaskedImageSrc(fetchImageUrl(filename, imageDir, 'masking') + '&t=0')
      }
    }).catch(() => {})
  }, [filename, imageDir])

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

  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        handleUndo()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleUndo])

  const handleBboxAdd = useCallback((coords) => {
    pushHistory()
    const id = crypto.randomUUID()
    setBboxes(prev => [...prev, { id, ...coords }])
    setSelectedId(id)
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

  const handleApply = async () => {
    if (bboxes.length === 0) return
    setApplying(true)
    setApplyStatus('')
    setApplyResult(null)
    try {
      const result = await applyMasking(filename, imageDir, bboxes, fontName || null)
      setApplyResult(result)
      setApplyStatus('success')
      // Cache-bust so the browser reloads the updated masked image
      setMaskedImageSrc(fetchImageUrl(result.output_filename, result.output_dir, 'masking') + `&t=${Date.now()}`)
    } catch (e) {
      console.error('Masking failed:', e)
      setApplyStatus('error')
    } finally {
      setApplying(false)
    }
  }

  const canUndo = bboxHistory.current.length > 0

  const btnBase = {
    padding: '6px 14px', borderRadius: '4px', fontSize: '14px', cursor: 'pointer',
  }

  const panelLabel = (text, color) => (
    <div style={{
      padding: '3px 10px', fontSize: '11px', fontWeight: 'bold',
      color, background: '#0d0a1a', borderBottom: `1px solid ${color}33`,
      userSelect: 'none', letterSpacing: '0.05em',
    }}>
      {text}
    </div>
  )

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 8px', borderBottom: '1px solid #2a1a5e', background: '#17102e',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            fontSize: '13px', color: '#c4aaff',
            borderLeft: '3px solid #7c4dff', paddingLeft: '8px',
          }}>
            マスキング
          </span>
          <span style={{ fontSize: '12px', color: '#7a6aaa' }}>{filename}</span>
        </div>

        <div style={{ display: 'flex', gap: '8px', padding: '6px 0', alignItems: 'center' }}>
          {fonts.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label style={{ fontSize: '12px', color: '#9a7acc' }}>フォント:</label>
              <select
                value={fontName}
                onChange={e => setFontName(e.target.value)}
                style={{
                  padding: '4px 8px', background: '#1b1a2e', color: '#e2eaf5',
                  border: '1px solid #3a2a6e', borderRadius: '4px', fontSize: '12px',
                }}
              >
                {fonts.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
              </select>
            </div>
          )}

          <button
            onClick={onBack}
            style={{ ...btnBase, border: '1px solid #3a2a6e', background: '#1b1a2e', color: '#a8aaee' }}
          >
            一覧へ
          </button>

          <button
            onClick={handleUndo}
            disabled={!canUndo}
            title="元に戻す (Ctrl+Z)"
            style={{
              ...btnBase,
              border: `1px solid ${canUndo ? '#3a2a6e' : '#2a1a4e'}`,
              background: canUndo ? '#1b1a2e' : '#131020',
              color: canUndo ? '#c4aaff' : '#4a3a7a',
              cursor: canUndo ? 'pointer' : 'not-allowed',
            }}
          >
            ↩ Undo
          </button>

          <button
            onClick={handleClearAll}
            disabled={bboxes.length === 0}
            style={{
              ...btnBase,
              background: bboxes.length === 0 ? '#2a1a4e' : '#5c0070',
              color: '#fff', border: 'none',
              cursor: bboxes.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            全消去
          </button>

          <button
            onClick={handleApply}
            disabled={bboxes.length === 0 || applying}
            style={{
              ...btnBase,
              background: applying ? '#3a1a6e' : bboxes.length === 0 ? '#2a1a4e' : '#5a2dff',
              color: bboxes.length === 0 && !applying ? '#5a4a9a' : '#fff',
              border: 'none', minWidth: '110px',
              cursor: bboxes.length === 0 || applying ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'center',
            }}
          >
            {applying && <span className="spinner" />}
            {applying ? '処理中...' : 'マスク適用'}
          </button>
        </div>
      </div>

      {/* Status banner */}
      {applyStatus === 'success' && applyResult && (
        <div style={{
          padding: '6px 16px', background: '#1a0a3a', borderBottom: '1px solid #5a2dff',
          fontSize: '12px', color: '#c4aaff',
          display: 'flex', alignItems: 'center', gap: '10px',
        }}>
          <span>✓ マスキング完了</span>
          <span style={{ fontFamily: 'monospace', background: '#0d0820', padding: '1px 6px', borderRadius: '3px' }}>
            inputs/{applyResult.output_dir}/{applyResult.output_filename}
          </span>
          <button
            onClick={() => setApplyStatus('')}
            style={{ background: 'none', border: 'none', color: '#5a4a8a', cursor: 'pointer', fontSize: '16px', marginLeft: 'auto' }}
          >
            ×
          </button>
        </div>
      )}
      {applyStatus === 'error' && (
        <div style={{
          padding: '6px 16px', background: '#2a0a1a', borderBottom: '1px solid #aa2255',
          fontSize: '12px', color: '#ff6688',
          display: 'flex', alignItems: 'center', gap: '10px',
        }}>
          <span>✗ マスキングに失敗しました</span>
          <button
            onClick={() => setApplyStatus('')}
            style={{ background: 'none', border: 'none', color: '#aa2255', cursor: 'pointer', fontSize: '16px', marginLeft: 'auto' }}
          >
            ×
          </button>
        </div>
      )}

      {/* Body: original | masked | sidebar */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Left: original image with bbox drawing */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
          {panelLabel('元画像', '#7c4dff')}
          <div style={{ flex: 1, overflow: 'hidden', background: '#0d1b2a' }}>
            <AnnotationCanvas
              imageSrc={filename ? fetchImageUrl(filename, imageDir) : ''}
              bboxes={bboxes}
              selectedId={selectedId}
              onBboxAdd={handleBboxAdd}
              onSelect={setSelectedId}
              bboxColor={BBOX_COLOR}
              externalZoom={sharedZoom}
              onZoomChange={setSharedZoom}
            />
          </div>
        </div>

        {/* Divider */}
        <div style={{ width: '2px', background: '#2a1a5e', flexShrink: 0 }} />

        {/* Right: masked result (read-only) */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
          {panelLabel('マスク済み', '#00b4aa')}
          <div style={{ flex: 1, overflow: 'hidden', background: '#0a1a18' }}>
            {maskedImageSrc ? (
              <AnnotationCanvas
                imageSrc={maskedImageSrc}
                bboxes={[]}
                selectedId={null}
                bboxColor={BBOX_COLOR}
                externalZoom={sharedZoom}
                onZoomChange={setSharedZoom}
                // onBboxAdd omitted → read-only
              />
            ) : (
              <div style={{
                width: '100%', height: '100%',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                color: '#2a4a48', fontSize: '13px', gap: '8px',
              }}>
                <span style={{ fontSize: '32px', opacity: 0.3 }}>⬜</span>
                <span>マスク未適用</span>
                <span style={{ fontSize: '11px', color: '#1a3a38' }}>
                  左でbboxを描いて「マスク適用」を押すと結果がここに表示されます
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div style={{
          width: '210px', flexShrink: 0,
          display: 'flex', flexDirection: 'column',
          background: '#110e22', borderLeft: '1px solid #2a1a5e',
        }}>
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid #2a1a5e',
            fontSize: '13px', color: '#7a5acc',
          }}>
            マスク領域 ({bboxes.length})
          </div>

          <div style={{ flex: 1, overflowY: 'auto' }}>
            {bboxes.length === 0 ? (
              <div style={{
                padding: '24px 12px', fontSize: '12px', color: '#3a2a5a',
                textAlign: 'center', lineHeight: 1.6,
              }}>
                画像上でドラッグして<br />マスク領域を追加
              </div>
            ) : (
              bboxes.map((b, i) => (
                <div
                  key={b.id}
                  onClick={() => setSelectedId(b.id)}
                  style={{
                    padding: '8px 12px',
                    borderBottom: '1px solid #1e1540',
                    background: selectedId === b.id ? '#1e1545' : 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    cursor: 'pointer', transition: 'background 0.1s',
                  }}
                  onMouseEnter={e => { if (selectedId !== b.id) e.currentTarget.style.background = '#16123a' }}
                  onMouseLeave={e => { if (selectedId !== b.id) e.currentTarget.style.background = 'transparent' }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      width: '20px', height: '20px', borderRadius: '50%',
                      background: '#3a1a8e', color: '#c4aaff',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '11px', fontWeight: 'bold', flexShrink: 0,
                    }}>
                      {i + 1}
                    </span>
                    <span style={{ fontSize: '11px', color: '#9a7acc' }}>
                      {Math.round(b.width * 100)}×{Math.round(b.height * 100)}%
                    </span>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(b.id) }}
                    style={{
                      background: 'none', border: 'none', color: '#5a3a8a',
                      cursor: 'pointer', fontSize: '14px', padding: '2px 4px',
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = '#ff6688'}
                    onMouseLeave={e => e.currentTarget.style.color = '#5a3a8a'}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>

          {bboxes.length > 0 && (
            <div style={{
              padding: '10px 12px', borderTop: '1px solid #2a1a5e',
              fontSize: '11px', color: '#5a4a8a', lineHeight: 1.5,
            }}>
              <div>フォント: {fontName || '(デフォルト)'}</div>
              <div>保存先: inputs/masking/{imageDir}/</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
