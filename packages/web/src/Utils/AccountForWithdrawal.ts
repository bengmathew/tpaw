export class AccountForWithdrawal {
  constructor(public balance: number) {}

  withdraw(x: number) {
    const amount = Math.min(x, this.balance)
    this.balance -= amount
    return amount
  }
}
