import React from 'react'
import { AppPage } from '../App/AppPage'

export const About = React.memo(() => {
  return (
    <AppPage title="About - TPAW Planner">
      <div className="">
        <h1 className="font-bold text-4xl ">About</h1>
        <p className="mt-6">
          TPAW was developed by Ben Mathew on the Bogleheads
          thread:{' '}
          <a
            className="underline"
            target="_blank"
            rel="noreferrer"
            href="https://www.bogleheads.org/forum/viewtopic.php?f=10&t=331368"
          >
            Total portfolio allocation and withdrawal (TPAW)
          </a>
          . 
          </p>
          <p className="mt-4">
          Software for this site was written by Ben’s brother, Jacob Mathew.
        </p>
      </div>
    </AppPage>
  )
})
