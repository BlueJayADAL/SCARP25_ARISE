import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import autoprefixer from 'autoprefixer'

export default defineConfig(() => {
  return {    build: {
      outDir: path.resolve(__dirname, '../server/static'),
    },
    css: {
      postcss: {
        plugins: [
          autoprefixer({}), // add options if needed
        ],
      },
    },
    esbuild: {
      loader: 'jsx',
      include: /src\/.*\.jsx?$/,
      exclude: [],
    },
    optimizeDeps: {
      force: true,
      esbuildOptions: {
        loader: {
          '.js': 'jsx',
        },
      },
    },
    plugins: [react()],
    resolve: {
      alias: [
        {
          find: 'src/',
          replacement: `${path.resolve(__dirname, 'src')}/`,
        },
      ],
      extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json', '.scss'],
    },
    server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  base: '/static/',
  }
})
