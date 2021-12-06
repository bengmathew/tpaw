import React from 'react'
import { AppPage } from '../App/AppPage'

export const Disclaimer = React.memo(() => {
  return (
    <AppPage title="Disclaimer - TPAW Planner">
      <div className="">
        <h1 className="font-bold text-4xl ">Disclaimer</h1>
        <p className="mt-6">
          The information provided on this website is for informational purposes
          only and should not be construed as financial advice. No warranty is
          made that the information provided on this website is accurate or
          appropriate for your circumstances. By using this website, you agree
          not to hold Benjamin Mathew or Jacob Mathew or any other entity
          associated with this website liable in any way for damages arising
          from decisions you make based on the information provided on this
          website.
        </p>
      </div>
    </AppPage>
  )
})
