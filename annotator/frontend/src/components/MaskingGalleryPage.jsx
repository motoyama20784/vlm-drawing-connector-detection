import { useState, useEffect } from 'react'
import { fetchDirs, fetchMaskingStatus, fetchImageUrl } from '../api.js'

function MaskBadge({ masked }) {
  if (masked) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#1a1a4a', color: '#7c4dff',
      }}>✓ マスク済み</span>
    )
  }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
      background: '#1e3a5f', color: '#7a9cc0',
    }}>未処理</span>
  )
}

export default function MaskingGalleryPage({ onSelectImage, selectedDir, onDirChange }) {
  const [dirs, setDirs] = useState([])
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchDirs().then(d => {
      setDirs(d)
      if (!selectedDir && d.length > 0) onDirChange(d[0])
    }).catch(console.error)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedDir) return
    setLoading(true)
    fetchMaskingStatus(selectedDir)
      .then(setImages)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedDir])

  const total = images.length
  const masked = images.filter(i => i.masked).length

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0d1b2a', minHeight: 0 }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px', flexShrink: 0,
        padding: '8px 16px', background: '#17102e', borderBottom: '1px solid #2a1a5e',
      }}>
        <span style={{
          fontWeight: 'bold', fontSize: '15px', color: '#c4aaff',
          borderLeft: '3px solid #7c4dff', paddingLeft: '10px',
        }}>
          マスキング
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '13px', color: '#9a7acc' }}>ディレクトリ:</label>
          <select
            value={selectedDir}
            onChange={e => onDirChange(e.target.value)}
            style={{
              padding: '4px 8px', background: '#1b1a2e', color: '#e2eaf5',
              border: '1px solid #3a2a6e', borderRadius: '4px', fontSize: '13px',
            }}
          >
            {dirs.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        {total > 0 && (
          <span style={{ fontSize: '13px', color: '#9a7acc' }}>
            {masked} / {total} マスク済み
            <span style={{
              display: 'inline-block', marginLeft: '8px',
              width: '80px', height: '6px', background: '#1e1a3f', borderRadius: '3px', verticalAlign: 'middle',
            }}>
              <span style={{
                display: 'block', height: '100%', borderRadius: '3px',
                background: '#7c4dff', width: `${(masked / total) * 100}%`,
                transition: 'width 0.3s',
              }} />
            </span>
          </span>
        )}
        {selectedDir && (
          <span style={{ fontSize: '12px', color: '#5a4a8a', marginLeft: 'auto' }}>
            保存先: inputs/masking/{selectedDir}/
          </span>
        )}
      </div>

      {/* Grid */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {loading ? (
          <div style={{ color: '#9a7acc', padding: '32px', textAlign: 'center' }}>読み込み中...</div>
        ) : images.length === 0 ? (
          <div style={{ color: '#4a3a6a', padding: '32px', textAlign: 'center' }}>画像が見つかりません</div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))',
            gap: '16px',
          }}>
            {images.map(img => (
              <div
                key={img.filename}
                onClick={() => onSelectImage(img.filename, selectedDir)}
                style={{
                  background: '#13102a', border: `1px solid ${img.masked ? '#3a2a7e' : '#1e1a3f'}`,
                  borderRadius: '8px', overflow: 'hidden', cursor: 'pointer',
                  transition: 'border-color 0.15s, transform 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = '#7c4dff'
                  e.currentTarget.style.transform = 'translateY(-2px)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = img.masked ? '#3a2a7e' : '#1e1a3f'
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                <div style={{ width: '100%', height: '180px', overflow: 'hidden', background: '#0a0a1a' }}>
                  <img
                    src={fetchImageUrl(img.filename, selectedDir)}
                    alt={img.filename}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    loading="lazy"
                  />
                </div>
                <div style={{ padding: '8px' }}>
                  <div style={{
                    fontSize: '11px', color: '#9a7acc', marginBottom: '4px',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {img.filename}
                  </div>
                  <MaskBadge masked={img.masked} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
