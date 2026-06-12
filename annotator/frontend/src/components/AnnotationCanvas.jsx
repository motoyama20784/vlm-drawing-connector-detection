import { useRef, useEffect, useState, useCallback } from 'react'

export default function AnnotationCanvas({ imageSrc, bboxes, onBboxAdd }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [drawing, setDrawing] = useState(false)
  const [startPos, setStartPos] = useState(null)
  const [currentPos, setCurrentPos] = useState(null)

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imageRef.current) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)

    bboxes.forEach((bbox, i) => {
      const x = (bbox.x_center - bbox.width / 2) * canvas.width
      const y = (bbox.y_center - bbox.height / 2) * canvas.height
      const w = bbox.width * canvas.width
      const h = bbox.height * canvas.height
      ctx.strokeStyle = '#00e676'
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = '#00e676'
      ctx.font = 'bold 13px sans-serif'
      ctx.fillText(String(i + 1), x + 3, y + 15)
    })

    if (drawing && startPos && currentPos) {
      ctx.strokeStyle = '#ff5252'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 3])
      const rx = Math.min(startPos.x, currentPos.x)
      const ry = Math.min(startPos.y, currentPos.y)
      const rw = Math.abs(currentPos.x - startPos.x)
      const rh = Math.abs(currentPos.y - startPos.y)
      ctx.strokeRect(rx, ry, rw, rh)
      ctx.setLineDash([])
    }
  }, [bboxes, drawing, startPos, currentPos])

  useEffect(() => {
    if (!imageSrc) return
    const img = new Image()
    img.onload = () => {
      imageRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      redraw()
    }
    img.src = imageSrc
  }, [imageSrc])

  useEffect(() => { redraw() }, [redraw])

  const getPos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    }
  }

  const handleMouseDown = (e) => {
    setDrawing(true)
    const pos = getPos(e)
    setStartPos(pos)
    setCurrentPos(pos)
  }

  const handleMouseMove = (e) => {
    if (!drawing) return
    setCurrentPos(getPos(e))
  }

  const handleMouseUp = (e) => {
    if (!drawing || !startPos) return
    const endPos = getPos(e)
    const canvas = canvasRef.current
    const x1 = Math.min(startPos.x, endPos.x)
    const y1 = Math.min(startPos.y, endPos.y)
    const x2 = Math.max(startPos.x, endPos.x)
    const y2 = Math.max(startPos.y, endPos.y)
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      onBboxAdd({
        x_center: (x1 + x2) / 2 / canvas.width,
        y_center: (y1 + y2) / 2 / canvas.height,
        width: (x2 - x1) / canvas.width,
        height: (y2 - y1) / canvas.height,
      })
    }
    setDrawing(false)
    setStartPos(null)
    setCurrentPos(null)
  }

  if (!imageSrc) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
        画像を選択してください
      </div>
    )
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ cursor: 'crosshair', maxWidth: '100%', maxHeight: '100%', display: 'block' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    />
  )
}
