import React from 'react'
import { Contentful } from '../../Utils/Contentful'
import { createContext } from '../../Utils/CreateContext'


export const WithPlanContent = React.memo(({className = ''}: {className?: string}) => {
  
  return <div className={`${className}`}></div>
})