import { useCallback, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { UploadCloud, ImageIcon, Loader2 } from 'lucide-react'

interface UploadCardProps {
  onFileSelected: (file: File) => void
  previewUrl: string | null
  isLoading: boolean
}

export function UploadCard({ onFileSelected, previewUrl, isLoading }: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files?.[0]
      if (file) onFileSelected(file)
    },
    [onFileSelected],
  )

  const isIdle = !isDragging && !previewUrl && !isLoading

  return (
    <Card className="group relative overflow-hidden border-white/10 bg-white/4 backdrop-blur-xl shadow-2xl shadow-black/40">
      <div className="pointer-events-none absolute inset-0 bg-linear-to-br from-sky-500/5 via-transparent to-purple-500/5 opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
      <CardContent className="relative">
        <motion.div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => !isLoading && inputRef.current?.click()}
          animate={{
            scale: isDragging ? 1.015 : 1,
            boxShadow: isIdle
              ? [
                  '0 0 0px rgba(56,189,248,0)',
                  '0 0 24px rgba(56,189,248,0.25)',
                  '0 0 0px rgba(56,189,248,0)',
                ]
              : '0 0 0px rgba(56,189,248,0)',
          }}
          transition={
            isIdle
              ? { boxShadow: { duration: 3, repeat: Infinity, ease: 'easeInOut' }, scale: { duration: 0.2 } }
              : { duration: 0.2 }
          }
          className={`relative flex flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer ${
            isDragging
              ? 'border-sky-400 bg-sky-400/10'
              : 'border-white/15 hover:border-white/30 hover:bg-white/5'
          }`}
        >
          <AnimatePresence mode="wait">
            {isLoading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center gap-3 py-4"
              >
                <Loader2 className="size-8 animate-spin text-sky-400" />
                <p className="text-sm text-white/70">running inference...</p>
              </motion.div>
            ) : previewUrl ? (
              <motion.img
                key="preview"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                src={previewUrl}
                alt="upload preview"
                className="max-h-64 rounded-lg object-contain"
              />
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col items-center gap-3"
              >
                <motion.div
                  animate={{ y: isDragging ? -4 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <UploadCloud className="size-10 text-white/40" />
                </motion.div>
                <p className="text-sm text-white/70">
                  drag and drop a chest x-ray, or click to browse
                </p>
                <p className="flex items-center gap-1 text-xs text-white/40">
                  <ImageIcon className="size-3" /> PNG or JPG
                </p>
              </motion.div>
            )}
          </AnimatePresence>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFileSelected(file)
            }}
          />
        </motion.div>
        {previewUrl && (
          <Button
            variant="secondary"
            className="mt-4 w-full"
            disabled={isLoading}
            onClick={() => inputRef.current?.click()}
          >
            {isLoading ? 'analyzing...' : 'choose a different image'}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
