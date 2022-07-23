import {GetStaticProps} from 'next'
import {Contentful} from '../../Utils/Contentful'
import {getMarketData, MarketData} from '../Common/GetMarketData'

type _FetchedInline = Awaited<ReturnType<typeof Contentful.fetchInline>>
type _Body = {body: _FetchedInline}
type _Intro = {intro: _FetchedInline}
type _Menu = {menu: _FetchedInline}
type _IntroAndBody = _Body & _Intro
type _IntroAndBodyAndMenu = _IntroAndBody & _Menu

export type PlanContent = {
  'age-and-retirement': {
    introRetired: _FetchedInline
    introNotRetired: _FetchedInline
    body: _FetchedInline
  }
  'current-portfolio-balance': _IntroAndBody
  'income-during-retirement': _IntroAndBody
  'future-savings': _IntroAndBody
  'extra-spending': _IntroAndBody
  'spending-ceiling-and-floor': _IntroAndBody
  legacy: {
    introAmount: _FetchedInline
    introAssets: _FetchedInline
    body: _FetchedInline
  }
  'stock-allocation': _IntroAndBody
  'spending-tilt': _IntroAndBody
  lmp: _IntroAndBody
  withdrawal: _IntroAndBody
  'compare-strategies': _IntroAndBody & {
    tpawIntro: _FetchedInline
    spawIntro: _FetchedInline
    sharpeRatioIntro: _FetchedInline
  }
  'expected-returns': _IntroAndBody
  inflation: _IntroAndBody
  simulation: {
    body: _FetchedInline
    introSampling: _FetchedInline
    introSamplingMonteCarlo: _FetchedInline
    introSamplingHistorical: _FetchedInline
  }
  dev: {body: _FetchedInline}
  chart: {
    spending: {
      total: _IntroAndBodyAndMenu
      regular: _IntroAndBodyAndMenu
      discretionary: _IntroAndBodyAndMenu
      essential: _IntroAndBodyAndMenu
    }
    portfolio: _IntroAndBodyAndMenu
    'asset-allocation-savings-portfolio': _IntroAndBodyAndMenu
    'asset-allocation-total-portfolio': _IntroAndBodyAndMenu
    withdrawalRate: _IntroAndBodyAndMenu
    sharpeRatio: _IntroAndBody
  }
}

export type PlanStaticProps = {
  marketData: MarketData
  content: PlanContent
}

export const planGetStaticProps: GetStaticProps<
  PlanStaticProps
> = async context => {
  const marketData = await getMarketData()

  return {
    props: {
      marketData,
      content: {
        'age-and-retirement': {
          introRetired: await Contentful.fetchInline('1dZPTbtQfLz3cyDrMGrAQB'),
          introNotRetired: await Contentful.fetchInline(
            '43EyTxBVHWOPcA6rgBpsnG'
          ),
          body: await Contentful.fetchInline('5EtkcdtSIg0rS8AETEsgnm'),
        },
        'current-portfolio-balance': {
          intro: await Contentful.fetchInline('3iLyyrQAhHnzuc4IdftWT3'),
          body: await Contentful.fetchInline('5RE7wTwvtTAsF1sWpKrFW2'),
        },
        'income-during-retirement': {
          intro: await Contentful.fetchInline('3OqUTPDVRGzgQcVkJV7Lew'),
          body: await Contentful.fetchInline('1MHvhL8ImdOL9FxE5qxK6F'),
        },
        'future-savings': {
          intro: await Contentful.fetchInline('2rPr5mMTcScftXhletDeb4'),
          body: await Contentful.fetchInline('5aJN2Z4tZ7zQ6Tw69VelRt'),
        },
        'extra-spending': {
          intro: await Contentful.fetchInline('01kv7sKzniBagrcIwX86tJ'),
          body: await Contentful.fetchInline('5zDvtk4dDOonIkoIyOeQH8'),
        },
        'spending-ceiling-and-floor': {
          intro: await Contentful.fetchInline('19Llaw2GVZhEfBTfGzE7Ns'),
          body: await Contentful.fetchInline('6hEbQkY7ctTpMpGV6fBBu2'),
        },
        legacy: {
          introAmount: await Contentful.fetchInline('aSdQuriQu9ztfs812MRJj'),
          introAssets: await Contentful.fetchInline('5glA8ryQcNh7SHP9ZlkZ2y'),
          body: await Contentful.fetchInline('5nCHpNy6ReAEtBQTvDTwBf'),
        },

        'stock-allocation': {
          body: await Contentful.fetchInline('3ofgPmJFLgtJpjl26E7jpB'),
          intro: await Contentful.fetchInline('xWXcgVScUfdK1PaTNQeKz'),
        },
        'spending-tilt': {
          body: await Contentful.fetchInline('6Dv02w4fUuFQUjyWxnR7Vq'),
          intro: await Contentful.fetchInline('4UwuCPjuTz3SbwUcZIrLEG'),
        },
        lmp: {
          body: await Contentful.fetchInline('3ofgPmJFLgtJpjl26E7jpB'),
          intro: await Contentful.fetchInline('5FiPQS04F4uFngEMJium3B'),
        },
        withdrawal: {
          body: await Contentful.fetchInline('7eGRhX0KpxK2wKCzDTHLOs'),
          intro: await Contentful.fetchInline('3H8rgiVzmnyD6H3ZUjvgp8'),
        },
        'compare-strategies': {
          intro: await Contentful.fetchInline('52f9yaDqUCBBg3mkqGdZPc'),
          tpawIntro: await Contentful.fetchInline('4qYue9K3cSpEkSrAhIn7AV'),
          spawIntro: await Contentful.fetchInline('5W26KpQeXY9nC3FgKioesF'),
          sharpeRatioIntro: await Contentful.fetchInline(
            '7wNIfORQHqumvZG6wWcmqG'
          ),
          body: await Contentful.fetchInline('5F0tZKpZ2SPvljHIGkPYmy'),
        },
        'expected-returns': {
          intro: await Contentful.fetchInline('2NxIclWQoxuk0TMVH0GjhR'),
          body: await Contentful.fetchInline('2GxHf6q4kfRrz6AnFLniFh'),
        },
        inflation: {
          intro: await Contentful.fetchInline('76BgIpwX9yZetMGungnfwC'),
          body: await Contentful.fetchInline('6LqbR3PBA1uDe9xU2V1hk9'),
        },
        simulation: {
          body: await Contentful.fetchInline('5alyO5geIHnQsw8ZMpbyf5'),
          introSampling: await Contentful.fetchInline('6EwnBJaN5FYISgyPdSQe7U'),
          introSamplingMonteCarlo: await Contentful.fetchInline(
            '4ysOynT6PPgYFnY1BaLuy5'
          ),
          introSamplingHistorical: await Contentful.fetchInline(
            '19JANDD3uLnVdh52HBMj6U'
          ),
        },
        dev: {
          body: await Contentful.fetchInline('6jDlL7lhj5dYAatVc8Grb7'),
        },
        chart: {
          spending: {
            total: {
              intro: await Contentful.fetchInline('6MH8oPq7ivMYJ4Ii8ZMtwg'),
              body: await Contentful.fetchInline('4tlCDgSlcKXfO8hfmZvBoF'),
              menu: await Contentful.fetchInline('21U0B92yz78PBR2YlX1CJP'),
            },
            regular: {
              intro: await Contentful.fetchInline('3e3y2gADSUszUxSqWvC2EV'),
              body: await Contentful.fetchInline('14KuXhGuaRok2dW0ppyArU'),
              menu: await Contentful.fetchInline('2tmRByrkc7dPNVOS63CUy3'),
            },
            discretionary: {
              intro: await Contentful.fetchInline('3pcwRLnI1BaBm0Hqv0Zyvg'),
              body: await Contentful.fetchInline('4QRDrEzc3uJHNI4SzRJ6r7'),
              menu: await Contentful.fetchInline('7F7JGrPs7ShQ2G4GyEkJ75'),
            },
            essential: {
              intro: await Contentful.fetchInline('3oWMTqbZARYjiZ2G9fuwTb'),
              body: await Contentful.fetchInline('4Vd9Ay5NnEodIMiBl83Vfs'),
              menu: await Contentful.fetchInline('7xqfxGtRc4pHZFJvR9zaY1'),
            },
          },
          portfolio: {
            intro: await Contentful.fetchInline('5KzxtC01WrdXnahFd98zet'),
            body: await Contentful.fetchInline('2iQeojVfV3Fw18WV4TRATR'),
            menu: await Contentful.fetchInline('7cRJH6TdeEPpcom84B8Dch'),
          },
          'asset-allocation-savings-portfolio': {
            intro: await Contentful.fetchInline('247ji3WlKEyiMcthzbcaUX'),
            body: await Contentful.fetchInline('1r6sBwyo6ulnzLQwQOM05L'),
            menu: await Contentful.fetchInline('4OsYFfjGHmGn8HucKv2thb'),
          },
          'asset-allocation-total-portfolio': {
            intro: await Contentful.fetchInline('1uowfKS8e8RqIANAucKTkX'),
            body: await Contentful.fetchInline('1k57vBTerLEOp5ADAxDxoV'),
            menu: await Contentful.fetchInline('8MHwQEcXljZlEIcZrQRKd'),
          },
          withdrawalRate: {
            intro: await Contentful.fetchInline('7nDVZLSFxZcdHWmuSqzo6o'),
            body: await Contentful.fetchInline('79KDyYdPfxwl7BceHPECCe'),
            menu: await Contentful.fetchInline('42bT4OaF5u9GXukOHXHnWz'),
          },

          // This is never show because it is displayed only on detail screen,
          // but it is here to keep the code simple and uniform. Resusing content
          // from above.
          sharpeRatio: {
            intro: await Contentful.fetchInline('7nDVZLSFxZcdHWmuSqzo6o'),
            body: await Contentful.fetchInline('79KDyYdPfxwl7BceHPECCe'),
          },
        },
      },
    },
  }
}
