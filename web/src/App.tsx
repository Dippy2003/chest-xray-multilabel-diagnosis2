import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { UploadCard } from '@/components/UploadCard'
import { ResultsPanel, type ClassPrediction } from '@/components/ResultsPanel'
import { GradCamPanel } from '@/components/GradCamPanel'
import { AuroraBackground } from '@/components/AuroraBackground'
import { Activity } from 'lucide-react'

interface PredictResponse {
  predictions: Record<string, ClassPrediction>
  gradcam: {
    class: string
    image_base64: string
  }
}

function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleFileSelected(file: File) {
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setIsLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/predict', { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`server returned ${response.status}`)
      const data: PredictResponse = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen text-white">
      <AuroraBackground />

      <motion.header
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="relative z-10 mx-auto flex max-w-6xl items-center justify-between px-6 pt-10"
      >
        <div className="flex items-center gap-3">
          <div className="relative flex size-10 items-center justify-center rounded-xl border border-white/10 bg-white/5 shadow-lg shadow-sky-500/10">
            <Activity className="size-5 text-sky-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">chest x-ray classifier</h1>
            <p className="text-sm text-white/40">
              multi-label disease detection · resnet50 · grad-cam
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/50 sm:flex">
          <span className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex size-2 rounded-full bg-emerald-400" />
          </span>
          model online
        </div>
      </motion.header>

      <main className="relative z-10 mx-auto grid max-w-6xl grid-cols-1 gap-6 px-6 py-10 lg:grid-cols-2 lg:items-start">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: 'easeOut' }}
          className="flex flex-col gap-6"
        >
          <UploadCard
            onFileSelected={handleFileSelected}
            previewUrl={previewUrl}
            isLoading={isLoading}
          />

          <AnimatePresence>
            {error && (
              <motion.p
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300"
              >
                {error}
              </motion.p>
            )}
          </AnimatePresence>
        </motion.div>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className="flex flex-col gap-6"
            >
              <ResultsPanel predictions={result.predictions} />
              {previewUrl && (
                <GradCamPanel
                  originalUrl={previewUrl}
                  gradcamBase64={result.gradcam.image_base64}
                  gradcamClass={result.gradcam.class}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
