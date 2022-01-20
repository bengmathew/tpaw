import React from 'react'

export default React.memo(() => {
  throw new Error('crash!')
  return <></>
})
