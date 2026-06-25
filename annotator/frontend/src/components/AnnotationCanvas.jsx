import { useRef, useEffect, useReducer, useCallback, useState } from 'react'

const COLOR_DEFAULT = '#00e676'
const COLOR_SELECTED = '#ff9800'
const ZOOM_MIN = 0.05
const ZOOM_MAX = 10
export const ZOOM_FACTOR = 1.15

export function zoomReducer(state, action) {
  switch (action.type) {
    case 'fit': {
      const { cw, ch, ww, wh } = action
      const s = Math.min(ww / cw, wh / ch, 1)
      return { scale: s, offset: { x: (ww - cw * s) / 2, y: (wh - ch * s) / 2 } }
    }
    case 'wheel': {
      const { mx, my, factor } = action
      const newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, state.scale * factor))
      return {
        scale: newScale,
        offset: {
          x: mx - (mx - state.offset.x) * (newScale / state.scale),
          y: my - (my - state.offset.y) * (newScale / state.scale),
        },
      }
    }
    case 'pan': {
      return { scale: state.scale, offset: { x: state.offset.x + action.dx, y: state.offset.y + action.dy } }
    }
    default:
      return state
  }
}

// externalZoom / onZoomChange を渡すと両キャンバスで zoom を共有できる。
// 省略時は内部 state で管理（既存の動作）。
export default function AnnotationCanvas({
  imageSrc, bboxes, selectedId, onBboxAdd, onSelect,
  bboxColor = COLOR_DEFAULT,
  externalZoom = null,    // 共有 zoom state（親が管理）
  onZoomChange = null,    // zoom が変わったときに新 state を通知するコールバック
}) {
  const wrapperRef = useRef(null)
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const drawingRef = useRef(false)
  const startPosRef = useRef(null)
  const currentPosRef = useRef(null)
  const panningRef = useRef(false)
  const panStartRef = useRef(null)
  const [panning, setPanning] = useState(false)

  // 内部 zoom（externalZoom が渡されない場合に使用）
  const [internalZoom, dispatchInternal] = useReducer(zoomReducer, { scale: 1, offset: { x: 0, y: 0 } })
  const zoom = externalZoom ?? internalZoom

  // zoomDispatch: 変更を常に最新 zoom から計算し、親か内部に通知
  const zoomRef = useRef(zoom)
  useEffect(() => { zoomRef.current = zoom }, [zoom])

  const zoomDispatch = useCallback((action) => {
    if (onZoomChange) {
      onZoomChange(zoomReducer(zoomRef.current, action))
    } else {
      dispatchInternal(action)
    }
  }, [onZoomChange])

  // externalZoom が既にセットされている場合は画像ロード時の自動 fit をスキップ
  const externalZoomRef = useRef(externalZoom)
  useEffect(() => { externalZoomRef.current = externalZoom }, [externalZoom])

  // --- draw ---
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imageRef.current) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)

    const lw = Math.max(1, canvas.width * 0.002)
    const fontSize = Math.max(12, Math.round(canvas.width * 0.009))

    bboxes.forEach((bbox, i) => {
      const isSelected = bbox.id === selectedId
      const x = (bbox.x_center - bbox.width / 2) * canvas.width
      const y = (bbox.y_center - bbox.height / 2) * canvas.height
      const w = bbox.width * canvas.width
      const h = bbox.height * canvas.height
      const color = isSelected ? COLOR_SELECTED : bboxColor

      if (isSelected) {
        ctx.fillStyle = 'rgba(255,152,0,0.15)'
        ctx.fillRect(x, y, w, h)
      }
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? lw * 1.5 : lw
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = color
      ctx.font = `bold ${fontSize}px sans-serif`
      ctx.fillText(String(i + 1), x + lw * 2, y + fontSize)
    })

    if (drawingRef.current && startPosRef.current && currentPosRef.current) {
      const sp = startPosRef.current
      const cp = currentPosRef.current
      ctx.strokeStyle = '#ff5252'
      ctx.lineWidth = lw
      ctx.setLineDash([lw * 4, lw * 2])
      ctx.strokeRect(
        Math.min(sp.x, cp.x), Math.min(sp.y, cp.y),
        Math.abs(cp.x - sp.x), Math.abs(cp.y - sp.y)
      )
      ctx.setLineDash([])
    }
  }, [bboxes, selectedId, bboxColor])

  useEffect(() => { redraw() }, [redraw])

  // --- image load → fit ---
  const fitToWrapper = useCallback(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    if (!canvas || !wrapper || !canvas.width) return
    zoomDispatch({ type: 'fit', cw: canvas.width, ch: canvas.height, ww: wrapper.clientWidth, wh: wrapper.clientHeight })
  }, [zoomDispatch])

  useEffect(() => {
    if (!imageSrc) return
    const img = new Image()
    img.onload = () => {
      imageRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      // 共有 zoom がすでにセットされていれば自動 fit しない（追従モード）
      if (!externalZoomRef.current) fitToWrapper()
      redraw()
    }
    img.src = imageSrc
  }, [imageSrc, fitToWrapper])

  // --- wheel zoom ---
  const handleWheel = useCallback((e) => {
    e.preventDefault()
    const wrapper = wrapperRef.current
    if (!wrapper) return
    const rect = wrapper.getBoundingClientRect()
    zoomDispatch({
      type: 'wheel',
      mx: e.clientX - rect.left,
      my: e.clientY - rect.top,
      factor: e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR,
    })
  }, [zoomDispatch])

  useEffect(() => {
    const wrapper = wrapperRef.current
    if (!wrapper) return
    wrapper.addEventListener('wheel', handleWheel, { passive: false })
    return () => wrapper.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  // --- canvas coordinate from mouse event ---
  const getPos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    }
  }

  const hitTest = useCallback((pos) => {
    const canvas = canvasRef.current
    for (let i = bboxes.length - 1; i >= 0; i--) {
      const b = bboxes[i]
      const x1 = (b.x_center - b.width / 2) * canvas.width
      const y1 = (b.y_center - b.height / 2) * canvas.height
      const x2 = (b.x_center + b.width / 2) * canvas.width
      const y2 = (b.y_center + b.height / 2) * canvas.height
      if (pos.x >= x1 && pos.x <= x2 && pos.y >= y1 && pos.y <= y2) return b.id
    }
    return null
  }, [bboxes])

  const stopPan = () => {
    panningRef.current = false
    panStartRef.current = null
    setPanning(false)
  }

  const handleMouseDown = (e) => {
    if (e.button === 2) {
      e.preventDefault()
      panningRef.current = true
      panStartRef.current = { x: e.clientX, y: e.clientY }
      setPanning(true)
      return
    }
    if (e.button !== 0) return
    e.preventDefault()
    const pos = getPos(e)
    const hit = hitTest(pos)
    if (hit) { onSelect?.(hit); return }
    onSelect?.(null)
    if (!onBboxAdd) return
    drawingRef.current = true
    startPosRef.current = pos
    currentPosRef.current = pos
    redraw()
  }

  const handleMouseMove = (e) => {
    if (panningRef.current && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.x
      const dy = e.clientY - panStartRef.current.y
      panStartRef.current = { x: e.clientX, y: e.clientY }
      zoomDispatch({ type: 'pan', dx, dy })
      return
    }
    if (!drawingRef.current) return
    currentPosRef.current = getPos(e)
    redraw()
  }

  const handleMouseUp = (e) => {
    if (e.button === 2) {
      stopPan()
      return
    }
    if (!drawingRef.current || !startPosRef.current) return
    const endPos = getPos(e)
    const canvas = canvasRef.current
    const x1 = Math.min(startPosRef.current.x, endPos.x)
    const y1 = Math.min(startPosRef.current.y, endPos.y)
    const x2 = Math.max(startPosRef.current.x, endPos.x)
    const y2 = Math.max(startPosRef.current.y, endPos.y)
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      onBboxAdd?.({
        x_center: (x1 + x2) / 2 / canvas.width,
        y_center: (y1 + y2) / 2 / canvas.height,
        width: (x2 - x1) / canvas.width,
        height: (y2 - y1) / canvas.height,
      })
    }
    drawingRef.current = false
    startPosRef.current = null
    currentPosRef.current = null
    redraw()
  }

  if (!imageSrc) {
    return (
      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#4a6a8a' }}>
        画像を選択してください
      </div>
    )
  }

  return (
    <div
      ref={wrapperRef}
      style={{ width: '100%', height: '100%', overflow: 'hidden', position: 'relative', background: '#0d1b2a' }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          transformOrigin: '0 0',
          transform: `translate(${zoom.offset.x}px, ${zoom.offset.y}px) scale(${zoom.scale})`,
          cursor: panning ? 'grabbing' : (onBboxAdd ? 'crosshair' : 'default'),
          display: 'block',
          imageRendering: zoom.scale > 2 ? 'pixelated' : 'auto',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={(e) => { stopPan(); handleMouseUp(e) }}
        onContextMenu={(e) => e.preventDefault()}
      />
      <div style={{
        position: 'absolute', bottom: 8, right: 8,
        display: 'flex', alignItems: 'center', gap: '6px',
        background: 'rgba(0,0,0,0.55)', borderRadius: '6px', padding: '3px 8px',
        fontSize: '12px', color: '#a0c0e0', userSelect: 'none',
      }}>
        <span>{Math.round(zoom.scale * 100)}%</span>
        <button
          onClick={fitToWrapper}
          title="ズームリセット"
          style={{
            background: 'none', border: '1px solid #2a4060', borderRadius: '4px',
            color: '#a0c0e0', cursor: 'pointer', fontSize: '11px', padding: '1px 6px',
          }}
        >
          Fit
        </button>
      </div>
    </div>
  )
}
