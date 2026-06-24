import { useState, useEffect, useCallback } from 'react'
import ResultsCanvas from './ResultsCanvas.jsx'
import { fetchImageUrl, fetchResultsEvaluate } from '../api.js'

function SectionList({ title, color, items }) {
  if (items.length === 0) return null
  return (
    <div style={{ marginBottom: '14px' }}>
      <div style={{
        fontSize: '11px', fontWeight: 'bold', color, marginBottom: '5px',
        borderBottom: `1px solid ${color}44`, paddingBottom: '3px',
      }}>
        {title} ({items.length})
      </div>
      {items.map((item, i) => (
        <div key={i} style={{
          fontSize: '12px', color: '#a0c0e0', padding: '2px 4px',
          background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.03)',
          borderRadius: '2px', fontFamily: 'monospace',
        }}>
          {item}
        </div>
      ))}
    </div>
  )
}

export default function ResultsViewer({ filename, dir, onBack }) {
  const [evaluation, setEvaluation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [iouThreshold, setIouThreshold] = useState(0.3)

  const doFetch = useCallback((file, iou) => {
    setLoading(true)
    fetchResultsEvaluate(dir, filename, file, iou)
      .then(data => setEvaluation(data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [dir, filename])

  useEffect(() => {
    setEvaluation(null)
    setSelectedFile(null)
    setIouThreshold(0.3)
    let cancelled = false
    setLoading(true)
    fetchResultsEvaluate(dir, filename, null, 0.3)
      .then(data => {
        if (cancelled) return
        setEvaluation(data)
        setSelectedFile(data.selected_result_file)
      })
      .catch(console.error)
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [filename, dir])

  const handleFileChange = (file) => {
    setSelectedFile(file)
    doFetch(file, iouThreshold)
  }

  const handleIouChange = (val) => {
    setIouThreshold(val)
    doFetch(selectedFile, val)
  }

  const metrics = evaluation?.matches?.metrics
  const tp = evaluation?.matches?.tp ?? []
  const nearFp = evaluation?.matches?.near_fp ?? []
  const ghostFp = evaluation?.matches?.ghost_fp ?? []
  const fn = evaluation?.matches?.fn ?? []

  const f1Color = !metrics ? '#7a9cc0'
    : metrics.f1 >= 0.8 ? '#00e676'
    : metrics.f1 >= 0.5 ? '#ffa726'
    : '#f44336'

  const fmt = v => Number.isInteger(v) ? String(v) : v.toFixed(3)

  const imageSrc = filename ? fetchImageUrl(filename, dir) : ''

  return (
    // flex:1 + min-height:0 ensures this fills the parent flex column without height:100% resolution issues
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0,
        padding: '6px 12px', background: '#0f1f10', borderBottom: '1px solid #1a3a1f',
      }}>
        <button
          onClick={onBack}
          style={{
            padding: '5px 12px', borderRadius: '4px', fontSize: '13px',
            border: '1px solid #2a5a2e', background: '#0d1a0e',
            color: '#a8e6a8', cursor: 'pointer',
          }}
        >
          ← 一覧
        </button>
        <span style={{
          fontSize: '13px', color: '#a8e6a8',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          flex: 1, minWidth: 0,
        }}>
          {filename}
        </span>
        {/* IoU threshold slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
          <label style={{ fontSize: '12px', color: '#6a9c6a', whiteSpace: 'nowrap' }}>IoU閾値:</label>
          <input
            type="range"
            min="0.1" max="0.9" step="0.01"
            value={iouThreshold}
            onChange={e => handleIouChange(Number(e.target.value))}
            style={{ width: '90px', accentColor: '#4caf50', cursor: 'pointer' }}
          />
          <span style={{
            fontSize: '12px', color: '#a8e6a8', fontFamily: 'monospace',
            minWidth: '30px', textAlign: 'right',
          }}>
            {iouThreshold.toFixed(2)}
          </span>
        </div>
        {evaluation?.result_files?.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
            <label style={{ fontSize: '12px', color: '#6a9c6a', whiteSpace: 'nowrap' }}>実験結果:</label>
            <select
              value={selectedFile ?? ''}
              onChange={e => handleFileChange(e.target.value)}
              style={{
                padding: '3px 6px', background: '#0f2015', color: '#e2eaf5',
                border: '1px solid #2a5a2e', borderRadius: '4px', fontSize: '12px',
                maxWidth: '300px',
              }}
            >
              {evaluation.result_files.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Canvas + Sidebar */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', overflow: 'hidden' }}>

        {/* Canvas area — position:relative so ResultsCanvas (position:absolute inset:0) fills it */}
        <div style={{ flex: 1, minWidth: 0, position: 'relative' }}>
          <ResultsCanvas imageSrc={imageSrc} evaluation={evaluation} />
          {/* Loading overlay — drawn on top of the canvas so the image stays mounted */}
          {loading && (
            <div style={{
              position: 'absolute', inset: 0, pointerEvents: 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(10,20,30,0.55)',
            }}>
              <span style={{ color: '#6a9c6a', fontSize: '14px' }}>評価データ読み込み中...</span>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div style={{
          width: '276px', flexShrink: 0, background: '#0f2015',
          borderLeft: '1px solid #1a3a1f', display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid #1a3a1f',
            fontSize: '12px', color: '#6a9c6a', flexShrink: 0,
          }}>
            評価結果
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
            {loading && !evaluation ? (
              <div style={{ color: '#3a6a3a', fontSize: '12px' }}>読み込み中...</div>
            ) : !metrics ? (
              <div style={{ color: '#3a6a3a', fontSize: '12px' }}>
                {evaluation?.result_files?.length === 0 ? '結果ファイルがありません' : 'GTが存在しません'}
              </div>
            ) : (
              <>
                {/* Big 3 metrics */}
                <div style={{ marginBottom: '12px', padding: '10px', background: '#0d1a0e', borderRadius: '6px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '10px' }}>
                    {[['F1', metrics.f1, f1Color], ['Precision', metrics.precision, '#c0d8f0'], ['Recall', metrics.recall, '#c0d8f0']].map(([label, val, color]) => (
                      <div key={label} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#6a9c6a' }}>{label}</div>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', color }}>{val.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ height: '1px', background: '#1a3a1f', margin: '4px 0 8px' }} />
                  <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '13px' }}>
                    <span style={{ color: '#4ade80' }}>TP: {metrics.tp}</span>
                    <span style={{ color: '#f87171' }}>FP: {metrics.fp}</span>
                    <span style={{ color: '#fb923c' }}>FN: {metrics.fn}</span>
                  </div>
                </div>

                {/* Detail metrics */}
                <div style={{ fontSize: '13px', marginBottom: '14px' }}>
                  <div style={{ color: '#6a9c6a', marginBottom: '6px', fontSize: '11px', fontWeight: 'bold' }}>詳細指標</div>
                  {[
                    ['見逃し率', fmt(metrics.miss_rate)],
                    ['平均IoU (TP)', fmt(metrics.avg_matched_iou)],
                  ].map(([label, val]) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid #1a3a1f' }}>
                      <span style={{ color: '#7a9cc0' }}>{label}</span>
                      <span style={{ color: '#c0d8f0', fontFamily: 'monospace' }}>{val}</span>
                    </div>
                  ))}
                </div>

                <SectionList
                  title="TP (正解)"
                  color="#60a5fa"
                  items={tp.map(p => `P${p.pred_idx + 1} ↔ GT${p.gt_idx + 1}   IoU: ${p.iou.toFixed(3)}`)}
                />
                <SectionList
                  title="Near FP (位置ズレ)"
                  color="#fb923c"
                  items={nearFp.map(p => `P${p.pred_idx + 1}   max IoU: ${p.max_iou_to_gt.toFixed(3)}`)}
                />
                <SectionList
                  title="Ghost FP (幻覚)"
                  color="#c084fc"
                  items={ghostFp.map(p => `P${p.pred_idx + 1}   max IoU: ${p.max_iou_to_gt.toFixed(3)}`)}
                />
                <SectionList
                  title="見逃し (FN)"
                  color="#f87171"
                  items={fn.map(p => `GT${p.gt_idx + 1}`)}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
