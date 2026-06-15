import { useState } from 'react'

const CATEGORY_OPTIONS = ['Male Connector', 'Female Connector']

function CopyRow({ label, value }) {
  const [copied, setCopied] = useState(false)
  const text = value ?? '—'
  const canCopy = value !== null && value !== undefined

  const handleCopy = (e) => {
    e.stopPropagation()
    if (!canCopy) return
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '4px', marginBottom: '2px' }}>
      <span style={{ flexShrink: 0 }}>{label}: <span style={{ color: '#e2eaf5' }}>{text}</span></span>
      <button
        onClick={handleCopy}
        disabled={!canCopy}
        title={copied ? 'コピー済み' : 'クリップボードにコピー'}
        style={{
          flexShrink: 0,
          background: copied ? '#1b5e20' : 'none',
          border: `1px solid ${copied ? '#00e676' : '#2a4060'}`,
          borderRadius: '3px',
          color: copied ? '#00e676' : '#5a8ab0',
          cursor: canCopy ? 'pointer' : 'default',
          fontSize: '10px',
          padding: '1px 5px',
          transition: 'all 0.2s',
          whiteSpace: 'nowrap',
        }}
      >
        {copied ? '✓' : '⎘'}
      </button>
    </div>
  )
}

const inputStyle = (isSelected) => ({
  width: '100%', marginTop: '2px', padding: '4px 6px',
  background: '#1b2d3e', color: '#e2eaf5',
  border: `1px solid ${isSelected ? '#ff9800' : '#2a4060'}`,
  borderRadius: '4px', fontSize: '13px',
})

export default function BboxList({ bboxes, selectedId, onSelect, onCategoryChange, onFieldChange, onInfer, onDelete, inferring }) {
  if (bboxes.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#4a6a8a', fontSize: '13px' }}>
        Canvas 上でドラッグして bbox を描いてください
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', flex: 1 }}>
      <datalist id="category-options">
        {CATEGORY_OPTIONS.map(c => <option key={c} value={c} />)}
      </datalist>

      {bboxes.map((bbox, i) => {
        const isSelected = bbox.id === selectedId
        return (
          <div
            key={bbox.id}
            onClick={() => onSelect(bbox.id)}
            style={{
              padding: '10px',
              borderBottom: '1px solid #1e3a5f',
              background: isSelected ? '#1a3a5c' : '#0f2035',
              borderLeft: isSelected ? '3px solid #ff9800' : '3px solid transparent',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontWeight: 'bold', color: isSelected ? '#ff9800' : '#00e676' }}>
                bbox-{i + 1}
              </span>
              <button
                onClick={e => { e.stopPropagation(); onDelete(bbox.id) }}
                style={{ background: 'none', border: 'none', color: '#f44336', cursor: 'pointer', fontSize: '16px' }}
              >×</button>
            </div>

            <div onClick={e => e.stopPropagation()}>
              <div style={{ marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', color: '#7a9cc0' }}>Category</label>
                <input
                  list="category-options"
                  value={bbox.category}
                  onChange={e => onCategoryChange(bbox.id, e.target.value)}
                  placeholder="Select or type..."
                  style={inputStyle(isSelected)}
                />
              </div>

              <div style={{ marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', color: '#7a9cc0' }}>Part Number</label>
                <input
                  type="text"
                  value={bbox.part_number ?? ''}
                  onChange={e => onFieldChange(bbox.id, 'part_number', e.target.value)}
                  placeholder="e.g. 12345-678"
                  style={inputStyle(isSelected)}
                />
              </div>

              <div style={{ marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', color: '#7a9cc0' }}>Notes</label>
                <input
                  type="text"
                  value={bbox.notes ?? ''}
                  onChange={e => onFieldChange(bbox.id, 'notes', e.target.value)}
                  placeholder="Optional notes"
                  style={inputStyle(isSelected)}
                />
              </div>
            </div>

            <button
              onClick={e => { e.stopPropagation(); onInfer(bbox) }}
              disabled={inferring === bbox.id}
              style={{
                width: '100%', padding: '5px',
                background: inferring === bbox.id ? '#2a3f55' : '#1a5276',
                color: '#e2eaf5', border: 'none', borderRadius: '4px',
                cursor: inferring === bbox.id ? 'not-allowed' : 'pointer',
                fontSize: '13px', marginBottom: '6px',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              }}
            >
              {inferring === bbox.id && <span className="spinner" />}
              {inferring === bbox.id ? '推論中...' : 'VLM 推論'}
            </button>

            {(bbox.vlm_text !== null || bbox.vlm_shape !== null) && (
              <div style={{ fontSize: '12px', color: '#7a9cc0', background: '#0a1826', padding: '6px', borderRadius: '4px' }}>
                <CopyRow label="Text" value={bbox.vlm_text} />
                <CopyRow label="Shape" value={bbox.vlm_shape} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
