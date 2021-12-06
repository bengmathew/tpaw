import Link from 'next/link'
import React from 'react'
import {AppPage} from '../App/AppPage'

export const License = React.memo(() => {
  return (
    <AppPage title="License - TPAW Planner">
      <div className="">
        <h1 className="font-bold text-4xl">License</h1>
        <h2 className="font-bold text-xl mt-6 ">Open Source MIT License</h2>

        <p className="mt-4">
          Copyright 2020-2021 Benjamin Mathew and Jacob Mathew
        </p>

        <p className="mt-4">
          {`Permission is hereby granted, free of charge, to any person obtaining
          a copy of this software and associated documentation files (the
          "Software"), to deal in the Software without restriction, including
          without limitation the rights to use, copy, modify, merge, publish,
          distribute, sublicense, and/or sell copies of the Software, and to
          permit persons to whom the Software is furnished to do so, subject to
          the following conditions:`}
        </p>

        <p className="mt-4">
          The above copyright notice and this permission notice shall be
          included in all copies or substantial portions of the Software.
        </p>

        <p className="mt-4">
          {`THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
          EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
          MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
          IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
          CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
          TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
          SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`}
        </p>
        <h2 className="font-bold text-xl mt-6">Source Code</h2>
        <p className="mt-4">
          The source code for this website is available on{' '}
          <Link href="https://github.com/bengmathew/tpawplanner">
            <a className="underline">GitHub</a>
          </Link>
          .
        </p>
      </div>
    </AppPage>
  )
})
