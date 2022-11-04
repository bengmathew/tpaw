import { toast } from 'react-toastify'

export const errorToast = (message = 'Something went wrong.') => {
  toast(message, { type: 'error' })
}
