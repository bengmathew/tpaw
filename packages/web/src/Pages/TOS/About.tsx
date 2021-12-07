import React from 'react'
import {AppPage} from '../App/AppPage'

export const About = React.memo(() => {
  return (
    <AppPage title="About - TPAW Planner">
      <div className="">
        <h1 className="font-bold text-4xl ">About</h1>
        <p className="mt-6">
          {`Total Portfolio Allocation and Withdrawal (TPAW) is a retirement
          strategy that uses the total portfolio approach to calculate asset
          allocation and withdrawal.`}
        </p>

        <p className="mt-6">
          {`The total portfolio approach means that the present value of future
          savings and retirement income, valued using the bond rate, is
          counted as bonds in the portfolio. A fixed asset allocation is
          maintained on this total portfolio. Retirement withdrawals are
          calculated by amortizing the total portfolio over retirement years
          (amortization based withdrawal).`}
        </p>

        <p className="mt-6">
          The advantage of the total portfolio approach is that total risk is
          kept consistent from year to year. This has two benefits:
          <ul className="list-decimal list-outside pl-5">
            <li className="mt-2">
              The more even spreading of risk across years reduces the total
              risk that the retiree would need to take to achieve a given
              expected return.
            </li>
            <li className="mt-2">
              It prevents surprises like risk increasing unexpectedly as the
              real value of a pension declines and the retiree relies more
              heavily on the savings portfolio.
            </li>
          </ul>
        </p>
        <p className="mt-6">
          TPAW was developed by Ben Mathew on the Bogleheads thread{' '}
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
          The software for this website was written by Ben’s brother, Jacob
          Mathew.
        </p>
      </div>
    </AppPage>
  )
})
