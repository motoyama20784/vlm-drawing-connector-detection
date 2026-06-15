import { useState, useEffect } from 'react'
import { fetchDirs, fetchStatus, fetchImageUrl } from '../api.js'

function StatusBadge({ completed, bboxCount }) {
  if (completed) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#0d3320', color: '#00e676',
      }}>✓ 完了 {bboxCount > 0 ? `(${bboxCount})` : ''}</span>
    )
  }
  if (bboxCount > 0) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#1a3a5c', color: '#5aabff',
      }}>進行中 ({bboxCount})</span>
    )
  }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
      background: '#1e3a5f', color: '#7a9cc0',
    }}>未</span>
  )
}

export default function GalleryPage({ onSelectImage }) {
  const [dirs, setDirs] = useState([])
  const [selectedDir, setSelectedDir] = useState('')
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchDirs().then(d => {
      setDirs(d)
      if (d.length > 0) setSelectedDir(d.includes('inputs') ? 'inputs' : d[0])
    }).catch(console.error)
  }, [])

  useEffect(() => {
    if (!selectedDir) return
    setLoading(true)
    fetchStatus(selectedDir)
      .then(setImages)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedDir])

  const total = images.length
  const done = images.filter(i => i.completed).length

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0d1b2a' }}>
      {/* ヘッダー */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px',
        padding: '8px 16px', background: '#112236', borderBottom: '1px solid #1e3a5f',
      }}>
        <span style={{ fontWeight: 'bold', fontSize: '15px', color: '#e2eaf5' }}>アノテーション一覧</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '13px', color: '#7a9cc0' }}>ディレクトリ:</label>
          <select
            value={selectedDir}
            onChange={e => setSelectedDir(e.target.value)}
            style={{
              padding: '4px 8px', background: '#1b2d3e', color: '#e2eaf5',
              border: '1px solid #2a4060', borderRadius: '4px', fontSize: '13px',
            }}
          >
            {dirs.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        {total > 0 && (
          <span style={{ fontSize: '13px', color: '#7a9cc0' }}>
            {done} / {total} 完了
            <span style={{
              display: 'inline-block', marginLeft: '8px',
              width: '80px', height: '6px', background: '#1e3a5f', borderRadius: '3px', verticalAlign: 'middle',
            }}>
              <span style={{
                display: 'block', height: '100%', borderRadius: '3px',
                background: '#00e676', width: `${total ? (done / total) * 100 : 0}%`,
                transition: 'width 0.3s',
              }} />
            </span>
          </span>
        )}
      </div>

      {/* グリッド */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {loading ? (
          <div style={{ color: '#7a9cc0', padding: '32px', textAlign: 'center' }}>読み込み中...</div>
        ) : images.length === 0 ? (
          <div style={{ color: '#4a6a8a', padding: '32px', textAlign: 'center' }}>画像が見つかりません</div>
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
                  background: '#112236', border: '1px solid #1e3a5f', borderRadius: '8px',
                  overflow: 'hidden', cursor: 'pointer',
                  transition: 'border-color 0.15s, transform 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = '#2a7aaf'
                  e.currentTarget.style.transform = 'translateY(-2px)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = '#1e3a5f'
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                <div style={{ width: '100%', height: '180px', overflow: 'hidden', background: '#0a1826' }}>
                  <img
                    src={fetchImageUrl(img.filename, selectedDir)}
                    alt={img.filename}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    loading="lazy"
                  />
                </div>
                <div style={{ padding: '8px' }}>
                  <div style={{
                    fontSize: '11px', color: '#7a9cc0', marginBottom: '4px',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {img.filename}
                  </div>
                  <StatusBadge completed={img.completed} bboxCount={img.bbox_count} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
