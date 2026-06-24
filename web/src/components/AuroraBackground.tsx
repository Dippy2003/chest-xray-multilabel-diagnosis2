import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

interface Node {
  x: number
  y: number
  vx: number
  vy: number
}

function NeuralField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let width = window.innerWidth
    let height = window.innerHeight
    let nodes: Node[] = []
    const NODE_COUNT = 70
    const LINK_DIST = 150

    function resize() {
      width = window.innerWidth
      height = window.innerHeight
      canvas!.width = width
      canvas!.height = height
    }

    function init() {
      nodes = Array.from({ length: NODE_COUNT }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
      }))
    }

    resize()
    init()
    window.addEventListener('resize', resize)

    let frame = 0
    let rafId: number

    function tick() {
      frame++
      ctx!.clearRect(0, 0, width, height)

      for (const n of nodes) {
        n.x += n.vx
        n.y += n.vy
        if (n.x < 0 || n.x > width) n.vx *= -1
        if (n.y < 0 || n.y > height) n.vy *= -1
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i]
          const b = nodes[j]
          const dx = a.x - b.x
          const dy = a.y - b.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < LINK_DIST) {
            const opacity = (1 - dist / LINK_DIST) * 0.35
            ctx!.strokeStyle = `rgba(56, 189, 248, ${opacity})`
            ctx!.lineWidth = 1
            ctx!.beginPath()
            ctx!.moveTo(a.x, a.y)
            ctx!.lineTo(b.x, b.y)
            ctx!.stroke()
          }
        }
      }

      for (const n of nodes) {
        const pulse = 1.4 + Math.sin(frame * 0.03 + n.x * 0.01) * 0.6
        ctx!.beginPath()
        ctx!.arc(n.x, n.y, pulse, 0, Math.PI * 2)
        ctx!.fillStyle = 'rgba(125, 211, 252, 0.8)'
        ctx!.fill()
      }

      rafId = requestAnimationFrame(tick)
    }

    tick()

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(rafId)
    }
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 opacity-80" />
}

export function AuroraBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#05060a]">
      <motion.div
        className="absolute -left-1/4 -top-1/4 size-[60vw] rounded-full bg-sky-500/40 blur-[100px]"
        animate={{ x: [0, 80, -40, 0], y: [0, 50, 90, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -right-1/4 top-1/3 size-[55vw] rounded-full bg-purple-500/35 blur-[100px]"
        animate={{ x: [0, -60, 40, 0], y: [0, 70, -40, 0] }}
        transition={{ duration: 24, repeat: Infinity, ease: 'easeInOut' }}
      />

      <NeuralField />

      <motion.div
        className="absolute inset-x-0 h-px bg-linear-to-r from-transparent via-sky-400/70 to-transparent"
        animate={{ top: ['0%', '100%'] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'linear' }}
      />

      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:48px_48px] mask-[radial-gradient(ellipse_70%_60%_at_50%_0%,black_40%,transparent_100%)]" />

    </div>
  )
}
