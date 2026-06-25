import { useRef, useCallback, useEffect } from 'react'

export function useBboxHistory(setBboxes, setSelectedId) {
  const historyRef = useRef([])
  const bboxesRef = useRef([])

  const syncRef = useCallback((bboxes) => {
    bboxesRef.current = bboxes
  }, [])

  const push = useCallback(() => {
    historyRef.current = [...historyRef.current.slice(-49), bboxesRef.current]
  }, [])

  const undo = useCallback(() => {
    if (historyRef.current.length === 0) return
    const prev = historyRef.current[historyRef.current.length - 1]
    historyRef.current = historyRef.current.slice(0, -1)
    setBboxes(prev)
    setSelectedId(null)
  }, [setBboxes, setSelectedId])

  const reset = useCallback(() => {
    historyRef.current = []
  }, [])

  const canUndo = () => historyRef.current.length > 0

  return { bboxesRef, syncRef, push, undo, reset, canUndo }
}
