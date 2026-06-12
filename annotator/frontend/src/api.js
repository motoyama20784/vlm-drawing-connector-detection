import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchImages = () =>
  api.get('/images').then(r => r.data.images)

export const fetchImageUrl = (filename) =>
  `/api/images/${encodeURIComponent(filename)}`

export const fetchAnnotation = (filename) =>
  api.get(`/annotations/${encodeURIComponent(filename)}`).then(r => r.data)

export const saveAnnotation = (filename, data) =>
  api.post(`/annotations/${encodeURIComponent(filename)}`, data).then(r => r.data)

export const inferBbox = (image, bbox) =>
  api.post('/infer', { image, bbox }).then(r => r.data)
