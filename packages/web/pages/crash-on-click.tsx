import { block } from '@tpaw/common'
import React from 'react'

export default React.memo(() => {
  return (
    <div className="page">
      <div className="flex gap-x-4 justify-center items-center h-screen">
        <button
          className="btn2-dark btn2-lg"
          onClick={() => {
            throw new Error('crash on click!')
          }}
        >
          onerror
        </button>
        <button
          className="btn2-dark btn2-lg"
          onClick={() => {
            void block(async () => {
              throw new Error('crash on click!')
            })
          }}
        >
          onunhandledrejection
        </button>
      </div>
    </div>
  )
})
