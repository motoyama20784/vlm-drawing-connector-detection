import { useRef, useEffect, useReducer, useCallback, useState } from 'react'
import { zoomReducer } from './AnnotationCanvas.jsx'

const C = {
  gtTp:      '#4ade80',
  gtFn:      '#facc15',
  gtFnZero:  '#ff1744',
  predTp:    '#60a5fa',
  nearFp:    '#fb923c',
  ghostFp:   '#c084fc',
}

function computeIou(a, b) {
  const ax1 = a.x_center - a.width / 2, ax2 = a.x_center + a.width / 2
  const ay1 = a.y_center - a.height / 2, ay2 = a.y_center + a.height / 2
  const bx1 = b.x_center - b.width / 2, bx2 = b.x_center + b.width / 2
  const by1 = b.y_center - b.height / 2, by2 = b.y_center + b.height / 2
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1))
  const inter = iw * ih
  if (inter === 0) return 0
  return inter / (a.width * a.height + b.width * b.height - inter)
}

export default function ResultsCanvas({ imageSrc, evaluation }) {
  const wrapperRef = useRef(null)
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const panningRef = useRef(false)
  const panStartRef = useRef(null)
  const [panning, setPanning] = useState(false)
  const [zoom, dispatchZoom] = useReducer(zoomReducer, { scale: 1, offset: { x: 0, y: 0 } })
  const zoomRef = useRef(zoom)
  useEffect(() => { zoomRef.current = zoom }, [zoom])

  // { type: 'gt'|'pred', idx: number } | null
  const [selectedBox, setSelectedBox] = useState(null)

  // Reset selection when switching images
  useEffect(() => { setSelectedBox(null) }, [imageSrc])

  const fitToWrapper = useCallback(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    if (!canvas || !wrapper || !canvas.width) return
    const ww = wrapper.clientWidth
    const wh = wrapper.clientHeight
    if (!ww || !wh) return
    dispatchZoom({ type: 'fit', cw: canvas.width, ch: canvas.height, ww, wh })
  }, [])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height

    ctx.clearRect(0, 0, W, H)
    ctx.drawImage(img, 0, 0, W, H)
    if (!evaluation) return

    const { gt_boxes = [], pred_boxes = [], matches = {} } = evaluation
    const { tp = [], near_fp = [] } = matches

    const tpGtSet = new Set(tp.map(p => p.gt_idx))
    const tpPredMap = new Map(tp.map(p => [p.pred_idx, p.iou]))
    const nearFpSet = new Set(near_fp.map(p => p.pred_idx))

    const lw = Math.max(2, W * 0.0018)
    const fontSize = Math.max(11, W * 0.009)

    function drawBox(box, color, dashed, label) {
      const x = (box.x_center - box.width / 2) * W
      const y = (box.y_center - box.height / 2) * H
      const w = box.width * W
      const h = box.height * H

      ctx.globalAlpha = 0.12
      ctx.fillStyle = color
      ctx.fillRect(x, y, w, h)
      ctx.globalAlpha = 1.0

      ctx.strokeStyle = color
      ctx.lineWidth = lw
      ctx.setLineDash(dashed ? [lw * 5, lw * 3] : [])
      ctx.strokeRect(x, y, w, h)
      ctx.setLineDash([])

      if (label) {
        ctx.font = `bold ${fontSize}px sans-serif`
        const tw = ctx.measureText(label).width
        const pad = fontSize * 0.3
        const lx = x + lw
        const ly = y >= fontSize + pad * 2 ? y : y + h + fontSize + pad
        ctx.fillStyle = 'rgba(8,16,28,0.82)'
        ctx.fillRect(lx - pad, ly - fontSize - pad * 0.5, tw + pad * 2, fontSize + pad)
        ctx.fillStyle = color
        ctx.fillText(label, lx, ly - pad * 0.5)
      }
    }

    gt_boxes.forEach((box, i) => {
      const sel = selectedBox?.type === 'gt' && selectedBox?.idx === i
      if (tpGtSet.has(i)) {
        drawBox(box, C.gtTp, true, sel ? `GT${i + 1}` : null)
      } else {
        const hasAnyOverlap = pred_boxes.some(p => computeIou(box, p) > 0)
        const color = hasAnyOverlap ? C.gtFn : C.gtFnZero
        drawBox(box, color, true, sel ? `GT${i + 1} 見逃し${hasAnyOverlap ? '' : '(完全)'}` : null)
      }
    })

    pred_boxes.forEach((box, i) => {
      const sel = selectedBox?.type === 'pred' && selectedBox?.idx === i
      if (tpPredMap.has(i)) {
        drawBox(box, C.predTp, false, sel ? `P${i + 1} IoU:${tpPredMap.get(i).toFixed(2)}` : null)
      } else if (nearFpSet.has(i)) {
        drawBox(box, C.nearFp, false, sel ? `P${i + 1} Near FP` : null)
      } else {
        drawBox(box, C.ghostFp, false, sel ? `P${i + 1} Ghost FP` : null)
      }
    })
  }, [evaluation, selectedBox])

  // Keep ref up-to-date so onload always calls the latest redraw without being a dep
  const redrawRef = useRef(redraw)
  redrawRef.current = redraw

  // Re-draw when evaluation or selection changes
  useEffect(() => { redraw() }, [redraw])

  // Load image only when imageSrc changes
  useEffect(() => {
    if (!imageSrc) return
    const img = new Image()
    img.onload = () => {
      imageRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      fitToWrapper()
      redrawRef.current()
    }
    img.onerror = () => console.error('Failed to load image:', imageSrc)
    img.src = imageSrc
  }, [imageSrc, fitToWrapper])

  const handleWheel = useCallback((e) => {
    e.preventDefault()
    const rect = wrapperRef.current?.getBoundingClientRect()
    if (!rect) return
    dispatchZoom({
      type: 'wheel',
      mx: e.clientX - rect.left,
      my: e.clientY - rect.top,
      factor: e.deltaY < 0 ? 1.15 : 1 / 1.15,
    })
  }, [])

  useEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  const handleClick = useCallback((e) => {
    // Ignore right-click
    if (e.button !== 0) return
    const wrapper = wrapperRef.current
    const canvas = canvasRef.current
    if (!wrapper || !canvas || !canvas.width || !canvas.height) return

    const rect = wrapper.getBoundingClientRect()
    const { offset, scale } = zoomRef.current
    const cx = (e.clientX - rect.left - offset.x) / scale
    const cy = (e.clientY - rect.top - offset.y) / scale
    const nx = cx / canvas.width
    const ny = cy / canvas.height

    const { pred_boxes = [], gt_boxes = [] } = evaluation ?? {}

    const hit = (box) =>
      nx >= box.x_center - box.width / 2 && nx <= box.x_center + box.width / 2 &&
      ny >= box.y_center - box.height / 2 && ny <= box.y_center + box.height / 2

    // Check pred boxes first (drawn on top)
    for (let i = pred_boxes.length - 1; i >= 0; i--) {
      if (hit(pred_boxes[i])) {
        setSelectedBox(prev => prev?.type === 'pred' && prev?.idx === i ? null : { type: 'pred', idx: i })
        return
      }
    }
    for (let i = gt_boxes.length - 1; i >= 0; i--) {
      if (hit(gt_boxes[i])) {
        setSelectedBox(prev => prev?.type === 'gt' && prev?.idx === i ? null : { type: 'gt', idx: i })
        return
      }
    }
    setSelectedBox(null)
  }, [evaluation])

  const stopPan = () => { panningRef.current = false; panStartRef.current = null; setPanning(false) }

  return (
    <div
      ref={wrapperRef}
      onClick={handleClick}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#0d1b2a', cursor: panning ? 'grabbing' : 'crosshair' }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', transformOrigin: '0 0',
          transform: `translate(${zoom.offset.x}px, ${zoom.offset.y}px) scale(${zoom.scale})`,
          imageRendering: zoom.scale > 2 ? 'pixelated' : 'auto',
        }}
        onMouseDown={e => {
          if (e.button !== 2) return
          e.preventDefault()
          panningRef.current = true
          panStartRef.current = { x: e.clientX, y: e.clientY }
          setPanning(true)
        }}
        onMouseMove={e => {
          if (!panningRef.current || !panStartRef.current) return
          const dx = e.clientX - panStartRef.current.x
          const dy = e.clientY - panStartRef.current.y
          panStartRef.current = { x: e.clientX, y: e.clientY }
          dispatchZoom({ type: 'pan', dx, dy })
        }}
        onMouseUp={stopPan}
        onMouseLeave={stopPan}
        onContextMenu={e => e.preventDefault()}
      />

      {/* Legend */}
      <div style={{
        position: 'absolute', top: 8, left: 8,
        background: 'rgba(8,16,28,0.85)', borderRadius: '6px', padding: '8px 12px',
        fontSize: '12px', lineHeight: '2', userSelect: 'none', pointerEvents: 'none',
      }}>
        {[
          [C.gtTp,     '┄┄', 'GT (TP)'],
          [C.gtFn,     '┄┄', 'GT (見逃し)'],
          [C.gtFnZero, '┄┄', 'GT (完全見逃し)'],
          [C.predTp,   '──', 'Pred TP'],
          [C.nearFp,   '──', 'Near FP'],
          [C.ghostFp,  '──', 'Ghost FP'],
        ].map(([color, line, label]) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color, fontFamily: 'monospace', fontSize: '14px' }}>{line}</span>
            <span style={{ color: '#c0d8f0' }}>{label}</span>
          </div>
        ))}
        <div style={{ marginTop: '4px', fontSize: '10px', color: '#3a5a7a' }}>クリックでラベル表示</div>
      </div>

      {/* Zoom indicator */}
      <div style={{
        position: 'absolute', bottom: 8, right: 8,
        display: 'flex', gap: '6px', alignItems: 'center',
        background: 'rgba(8,16,28,0.75)', borderRadius: '6px', padding: '3px 8px',
        fontSize: '12px', color: '#a0c0e0', userSelect: 'none',
      }}>
        <span>{Math.round(zoom.scale * 100)}%</span>
        <button
          onClick={e => { e.stopPropagation(); fitToWrapper() }}
          style={{
            background: 'none', border: '1px solid #2a4060', borderRadius: '4px',
            color: '#a0c0e0', cursor: 'pointer', fontSize: '11px', padding: '1px 6px',
          }}
        >Fit</button>
      </div>
    </div>
  )
}
