import gsap from 'gsap'
import CustomEase from 'gsap/dist/CustomEase'
// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
gsap.registerPlugin(CustomEase)

export const cssEaseAsGSAPEase = CustomEase.create(
  'cssEase',
  '0.25,0.1,0.25,1.0'
)
