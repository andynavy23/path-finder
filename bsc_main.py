import time
import threading
import json
from web3 import Web3
from decimal import Decimal
from web3 import HTTPProvider
from web3._utils.request import make_post_request
from eth_abi import decode_abi


tokenIn = {
    'address': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',
    'symbol': 'WBNB',
    'decimal': 18,
}
tokenOut = tokenIn
startToken = tokenIn
currentPairs = []
path = [tokenIn]
bestTrades = []
maxHops = 6

bsc_url = 'https://bsc-dataseed.binance.org/'
w3 = Web3(HTTPProvider(bsc_url, request_kwargs={'timeout': 6000}))

pancakeswap_abi_path = './PancakeSwapPair.json'
pairABI = json.load(open(pancakeswap_abi_path))

pancakeswap_pairs_path = './pancakeswap_pairs.json'

d998 = Decimal(998)
d1000 = Decimal(1000)


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print('thread exception:', e)
            return None


class BatchHTTPProvider(HTTPProvider):

    def make_batch_request(self, text):
        self.logger.debug("Making request HTTP. URI: %s, Request: %s",
                          self.endpoint_uri, text)
        request_data = text.encode('utf-8')
        raw_response = make_post_request(
            self.endpoint_uri,
            request_data,
            **self.get_request_kwargs()
        )
        response = self.decode_rpc_response(raw_response)
        self.logger.debug("Getting response HTTP. URI: %s, "
                          "Request: %s, Response: %s",
                          self.endpoint_uri, text, response)
        return response


batch_provider = BatchHTTPProvider(bsc_url, request_kwargs={'timeout': 6000})


def toDict(pairs):
    p = {}
    i = 0
    for pair in pairs:
        p[pair['address']] = pair
        p[pair['address']]['arrIndex'] = i
        i += 1
    return p


def selectPairs(all_pairs):
    pairsDict = toDict(all_pairs)
    return all_pairs, pairsDict


def rpc_response_to_result(response):
    result = response.get('result')
    if result is None:
        error_message = 'result is None in response {}.'.format(response)
        if response.get('error') is None:
            error_message = error_message + ' Make sure Ethereum node is synced.'
            # When nodes are behind a load balancer it makes sense to retry the request in hopes it will go to other,
            # synced node
            raise ValueError(error_message)
        elif response.get('error') is not None and ValueError(response.get('error').get('code')):
            raise ValueError(error_message)
        raise ValueError(error_message)
    return result


def generate_json_rpc(method, params, request_id=1):
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
        'id': request_id,
    }


def generate_get_reserves_json_rpc(pairs, blockNumber='latest'):
    c = w3.eth.contract(abi=pairABI)
    for pair in pairs:
        yield generate_json_rpc(
                method='eth_call',
                params=[{
                    'to': pair['address'],
                    'data': c.encodeABI(fn_name='getReserves', args=[]),
                    },
                    hex(blockNumber) if blockNumber != 'latest' else 'latest',
                    ]
                )


def rpc_response_batch_to_results(response):
    for response_item in response:
        yield rpc_response_to_result(response_item)


def get_reserves(pairs, blockNumber='latest'):
    r = list(generate_get_reserves_json_rpc(pairs, blockNumber))
    resp = batch_provider.make_batch_request(json.dumps(r))
    results = list(rpc_response_batch_to_results(resp))
    for i in range(len(results)):
        res = decode_abi(['uint256', 'uint256', 'uint256'], bytes.fromhex(results[i][2:]))
        pairs[i]['reserve0'] = res[0]
        pairs[i]['reserve1'] = res[1]
    return pairs


def get_reserves_batch_mt(pairs):
    if len(pairs) <= 100:
        new_pairs = get_reserves(pairs)
    else:
        s = 0
        threads = []
        while s < len(pairs):
            e = s + 100
            if e > len(pairs):
                e = len(pairs)
            t = MyThread(func=get_reserves, args=(pairs[s:e],))
            t.start()
            threads.append(t)
            s = e
        new_pairs = []
        for t in threads:
            t.join()
            ret = t.get_result()
            new_pairs.extend(ret)
    return new_pairs


def getOptimalAmount(Ea, Eb):
    if Ea > Eb:
        return None
    if not isinstance(Ea, Decimal):
        Ea = Decimal(Ea)
    if not isinstance(Eb, Decimal):
        Eb = Decimal(Eb)
    return Decimal(int((Decimal.sqrt(Ea * Eb * d998 * d1000) - Ea * d1000) / d998))


def getAmountOut(amountIn, reserveIn, reserveOut):
    assert amountIn > 0
    assert reserveIn > 0 and reserveOut > 0
    if not isinstance(amountIn, Decimal):
        amountIn = Decimal(amountIn)
    if not isinstance(reserveIn, Decimal):
        reserveIn = Decimal(reserveIn)
    if not isinstance(reserveOut, Decimal):
        reserveOut = Decimal(reserveOut)
    return d998 * amountIn * reserveOut / (d1000 * reserveIn + d998 * amountIn)


def getEaEb(tokenIn, pairs):
    def adjustReserve(token, amount):
        # res = Decimal(amount)*Decimal(pow(10, 18-token['decimal']))
        # return Decimal(int(res))
        return amount

    def toInt(n):
        return Decimal(int(n))

    Ea = None
    Eb = None
    idx = 0
    tokenOut = tokenIn.copy()
    for pair in pairs:
        if idx == 0:
            if tokenIn['address'] == pair['token0']['address']:
                tokenOut = pair['token1']
            else:
                tokenOut = pair['token0']
        if idx == 1:
            Ra = adjustReserve(pairs[0]['token0'], pairs[0]['reserve0'])
            Rb = adjustReserve(pairs[0]['token1'], pairs[0]['reserve1'])
            if tokenIn['address'] == pairs[0]['token1']['address']:
                temp = Ra
                Ra = Rb
                Rb = temp
            Rb1 = adjustReserve(pair['token0'], pair['reserve0'])
            Rc = adjustReserve(pair['token1'], pair['reserve1'])
            if tokenOut['address'] == pair['token1']['address']:
                temp = Rb1
                Rb1 = Rc
                Rc = temp
                tokenOut = pair['token0']
            else:
                tokenOut = pair['token1']
            Ea = toInt(d1000 * Ra * Rb1 / (d1000 * Rb1 + d998 * Rb))
            Eb = toInt(d998 * Rb * Rc / (d1000 * Rb1 + d998 * Rb))
        if idx > 1:
            Ra = Ea
            Rb = Eb
            Rb1 = adjustReserve(pair['token0'], pair['reserve0'])
            Rc = adjustReserve(pair['token1'], pair['reserve1'])
            if tokenOut['address'] == pair['token1']['address']:
                temp = Rb1
                Rb1 = Rc
                Rc = temp
                tokenOut = pair['token0']
            else:
                tokenOut = pair['token1']
            Ea = toInt(d1000 * Ra * Rb1 / (d1000 * Rb1 + d998 * Rb))
            Eb = toInt(d998 * Rb * Rc / (d1000 * Rb1 + d998 * Rb))
        idx += 1
    return Ea, Eb


def findArb(pairs, tokenIn, tokenOut, maxHops, currentPairs, path, bestTrades, count=5):
    def sortTrades(trades, newTrade):
        trades.append(newTrade)
        return sorted(trades, key=lambda x: x['profit'])

    for i in range(len(pairs)):
        newPath = path.copy()
        pair = pairs[i]
        if not pair['token0']['address'] == tokenIn['address'] and not pair['token1']['address'] == tokenIn['address']:
            continue
        if pair['reserve0']/pow(10, pair['token0']['decimal']) < 1 or pair['reserve1']/pow(10, pair['token1']['decimal']) < 1:
            continue
        if tokenIn['address'] == pair['token0']['address']:
            tempOut = pair['token1']
        else:
            tempOut = pair['token0']
        newPath.append(tempOut)
        if tempOut['address'] == tokenOut['address'] and len(path) > 2:
            Ea, Eb = getEaEb(tokenOut, currentPairs + [pair])
            newTrade = { 'route': currentPairs + [pair], 'path': newPath, 'Ea': Ea, 'Eb': Eb }
            if Ea and Eb and Ea < Eb:
                newTrade['optimalAmount'] = getOptimalAmount(Ea, Eb)
                if newTrade['optimalAmount'] > 0:
                    newTrade['outputAmount'] = getAmountOut(newTrade['optimalAmount'], Ea, Eb)
                    newTrade['profit'] = newTrade['outputAmount']-newTrade['optimalAmount']
                    newTrade['p'] = int(newTrade['profit'])/pow(10, tokenOut['decimal'])
                else:
                    continue
                bestTrades = sortTrades(bestTrades, newTrade)
                bestTrades.reverse()
                bestTrades = bestTrades[:count]
        elif maxHops > 1 and len(pairs) > 1:
            pairsExcludingThisPair = pairs[:i] + pairs[i+1:]
            bestTrades = findArb(pairsExcludingThisPair, tempOut, tokenOut, maxHops-1, currentPairs + [pair], newPath, bestTrades, count)
    return bestTrades


def get_gas_fees():
    gas_limit = 70000
    gas_price = 10
    gas_fees = gas_price * gas_limit * Decimal('0.000000001')
    return gas_fees


def show_message(trades):
    gasFees = get_gas_fees()
    print('')
    text = ''
    for data in trades:
        print(data)
        input_amount = data['optimalAmount'] / pow(10, data['path'][0]['decimal'])
        symbols = [i['symbol'] for i in data['path']]
        output_amount = data['outputAmount'] / pow(10, data['path'][0]['decimal'])
        profit = data['profit'] / pow(10, data['path'][0]['decimal'])
        pnl = profit - gasFees
        usdt_input_amount = input_amount * Decimal('326.67')
        usdt_output_amount = output_amount * Decimal('326.67')
        usdt_pnl = pnl * Decimal('326.67')
        usdt_profit = profit * Decimal('326.67')

        if profit >= 0:
            text += f"Input amount: {round(input_amount, 4)} WBNB ({round(usdt_input_amount, 4)} USDT)\n"
            text += 'Path: ' + ' >> '.join(symbols) + '\n'
            text += f"Output amount: {round(output_amount, 4)} WBNB ({round(usdt_output_amount, 4)} USDT)\n"
            text += f"Profit(include transaction fee): {round(profit, 4)} WBNB ({round(usdt_profit, 4)} USDT)\n"
            text += f"Pnl: {round(pnl, 4)} WBNB ({round(usdt_pnl, 4)} USDT)\n\n"

    if text != '':
        print(text)


def main():
    start = time.time()
    all_pairs = json.load(open(pancakeswap_pairs_path))
    pairs, pairsDict = selectPairs(all_pairs)
    print('pairs:', len(pairs))

    try:
        pairs = get_reserves_batch_mt(pairs)
    except Exception as e:
        print('get_reserves err:', e)
        return
    end = time.time()
    print('update cost:', end - start, 's')

    trades = findArb(pairs, tokenIn, tokenOut, maxHops, currentPairs, path, bestTrades)
    if len(trades) == 0:
        print('No trades.')
        return

    show_message(trades)
    exit()


if __name__ == "__main__":
    main()
