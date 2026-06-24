import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export interface ClassPrediction {
  probability: number
  predicted: boolean
}

interface ResultsPanelProps {
  predictions: Record<string, ClassPrediction>
}

export function ResultsPanel({ predictions }: ResultsPanelProps) {
  const entries = Object.entries(predictions).sort(
    (a, b) => b[1].probability - a[1].probability,
  )

  return (
    <Card className="border-white/10 bg-white/4 backdrop-blur-xl shadow-2xl shadow-black/40">
      <CardHeader>
        <CardTitle className="text-white/90">predictions</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        {entries.map(([className, { probability, predicted }], i) => (
          <motion.div
            key={className}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: i * 0.08, ease: 'easeOut' }}
            className="flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-white/80">{className}</span>
              <div className="flex items-center gap-2">
                <motion.span
                  className="tabular-nums text-white/50"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.08 + 0.3 }}
                >
                  {(probability * 100).toFixed(1)}%
                </motion.span>
                {predicted && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.7 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.08 + 0.35, type: 'spring', stiffness: 300, damping: 18 }}
                  >
                    <Badge variant="default" className="bg-sky-500/90 text-white">
                      positive
                    </Badge>
                  </motion.div>
                )}
              </div>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
              <motion.div
                className={`h-full rounded-full ${
                  predicted
                    ? 'bg-linear-to-r from-sky-400 to-sky-500'
                    : 'bg-linear-to-r from-white/30 to-white/40'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${probability * 100}%` }}
                transition={{ duration: 0.7, delay: i * 0.08 + 0.1, ease: 'easeOut' }}
              />
            </div>
          </motion.div>
        ))}
      </CardContent>
    </Card>
  )
}
