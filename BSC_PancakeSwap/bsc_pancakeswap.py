import sys
sys.path.append('/home/ubuntu/uniswap-arbitrage-analysis')
import time
import threading
import json
from web3 import Web3
from decimal import Decimal
from web3 import HTTPProvider
from web3._utils.request import make_post_request
from eth_abi import decode_abi
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Provider
bsc_url = 'https://bsc-dataseed.binance.org/'

# Start token
tokenIn = {
    'address': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',
    'symbol': 'WBNB',
    'decimal': 18,
}

# About find path setting
maxHops = 6
num_trades = 1
allow_slippage_percent = Decimal('0.005')
tokenOut = tokenIn
startToken = tokenIn
currentPairs = []
path = [tokenIn]
bestTrades = []
w3 = Web3(HTTPProvider(bsc_url, request_kwargs={'timeout': 6000}))

# About DEX exchange setting
pancakeswap_abi_path = './abi/PancakeSwapPair_V2.json'
pairABI = json.load(open(pancakeswap_abi_path))
pancakeswap_pairs_path = './pancakeswap_pairs_V2.json'
pancakeswap_blacklist_path = './pancakeswap_blacklist_V2.json'

# About transaction fees calculation
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


def removeBlackList(pairs):
    blacklist = json.load(open(pancakeswap_blacklist_path))
    r = []
    for i in range(len(pairs)):
        if pairs[i]['token0']['address'] in blacklist or pairs[i]['token1']['address'] in blacklist:
            r.append(i)
    r.reverse()
    for t in r:
        del pairs[t]
    return pairs


def selectPairs(all_pairs):
    all_pairs = removeBlackList(all_pairs)
    return all_pairs


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
    if len(pairs) <= 200:
        new_pairs = get_reserves(pairs)
    else:
        s = 0
        threads = []
        while s < len(pairs):
            e = s + 200
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
            Ra = pairs[0]['reserve0']
            Rb = pairs[0]['reserve1']
            if tokenIn['address'] == pairs[0]['token1']['address']:
                temp = Ra
                Ra = Rb
                Rb = temp
            Rb1 = pair['reserve0']
            Rc = pair['reserve1']
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
            Rb1 = pair['reserve0']
            Rc = pair['reserve1']
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
            newTrade = {'route': currentPairs + [pair], 'path': newPath, 'Ea': Ea, 'Eb': Eb}
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
    gas_limit = 600000
    gas_price = 5
    gas_fees = gas_price * gas_limit * Decimal('0.000000001')
    return gas_fees


def get_output_amount(data, input_amount, slippage_percent=Decimal('1')):
    # print(data)
    path = [i['symbol'] for i in data['path']]

    pairs_info = []
    for index, item in enumerate(data['route']):
        if index == len(data['route']):
            continue
        token_f = path[index]
        token_b = path[index + 1]
        reserve_f = item['reserve0'] if item['token0']['symbol'] == token_f else item['reserve1']
        decimal_f = item['token0']['decimal'] if item['token0']['symbol'] == token_f else item['token1']['decimal']
        reserve_b = item['reserve1'] if item['token1']['symbol'] == token_b else item['reserve0']
        decimal_b = item['token1']['decimal'] if item['token1']['symbol'] == token_b else item['token0']['decimal']

        pairs_info.append({
            'pair_name': f"{token_f}/{token_b}",
            'token_f': token_f,
            'token_b': token_b,
            'reserve_f': reserve_f,
            'reserve_b': reserve_b,
            'decimal_f': decimal_f,
            'decimal_b': decimal_b,
        })

    # capital = input_amount / pow(10, data['path'][0]['decimal'])
    capital_reserve = input_amount
    for pair in pairs_info:
        update_capital_reserve = getAmountOut(capital_reserve, pair['reserve_f'], pair['reserve_b'])
        # update_capital = update_capital_reserve / pow(10, pair['decimal_b'])
        # print(f"Buy {pair['pair_name']} with {capital} {pair['token_f']}")
        # print(f"Get {update_capital} {pair['token_b']} back")
        # capital = update_capital
        capital_reserve = update_capital_reserve

    return capital_reserve - (capital_reserve * slippage_percent)


def send_message(trades, num_trades: int = 1, test_mode: bool = False):
    gasFees = get_gas_fees()
    texts = []
    update_trades = []
    for data in trades:
        text = ''
        # print(data)
        input_amount = data['optimalAmount']
        output_amount = data['outputAmount']
        output_slippage_amount = get_output_amount(data, input_amount, allow_slippage_percent)
        input_amount = input_amount / pow(10, data['path'][0]['decimal'])
        output_amount = output_amount / pow(10, data['path'][0]['decimal'])
        output_slippage_amount = output_slippage_amount / pow(10, data['path'][0]['decimal'])
        symbols = [i['symbol'] for i in data['path']]

        profit = (output_amount - input_amount)
        profit_slippage = (output_slippage_amount - input_amount)
        pnl = profit - gasFees
        pnl_slippage = profit_slippage - gasFees

        usdt_input_amount = input_amount * Decimal('326.67')
        usdt_output_amount = output_amount * Decimal('326.67')
        usdt_output_slippage_amount = output_slippage_amount * Decimal('326.67')
        usdt_profit = profit * Decimal('326.67')
        usdt_profit_slippage = profit_slippage * Decimal('326.67')
        usdt_pnl = pnl * Decimal('326.67')
        usdt_pnl_slippage = pnl_slippage * Decimal('326.67')

        logger.info(f"Path: {' >> '.join([i['symbol'] for i in data['path']])}")
        logger.info(f"Input amount: {round(input_amount, 4)} WBNB({round(usdt_input_amount, 4)} USDT)")
        logger.info(f"Output amount: {round(output_amount, 4)} WBNB({round(usdt_output_amount, 4)} USDT)")
        logger.info(f"Output amount(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(output_slippage_amount, 4)} WBNB({round(usdt_output_slippage_amount, 4)} USDT)")
        logger.info(f"Profit: {round(profit, 4)} WBNB({round(usdt_profit, 4)} USDT)")
        logger.info(f"Profit(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(profit_slippage, 4)} WBNB({round(usdt_profit_slippage, 4)} USDT)")
        logger.info(f"Pnl: {round(pnl, 4)} WBNB({round(usdt_pnl, 4)} USDT)")
        logger.info(f"Pnl(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(pnl_slippage, 4)} WBNB({round(usdt_pnl_slippage, 4)} USDT)")
        logger.info(f"Trade condition: {round(input_amount * Decimal('0.001'), 4)} WBNB({round(usdt_input_amount * Decimal('0.001'), 4)} USDT)")

        if pnl_slippage > 0:
            text += f"Input amount: {round(input_amount, 4)} WBNB ({round(usdt_input_amount, 4)} USDT)\n"
            text += 'Path: ' + ' >> '.join(symbols) + '\n'
            text += f"Output amount: {round(output_amount, 4)} WBNB ({round(usdt_output_amount, 4)} USDT)\n"
            text += f"Output amount(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(output_slippage_amount, 4)} WBNB ({round(usdt_output_slippage_amount, 4)} USDT)\n"
            text += f"Profit: {round(profit, 4)} WBNB ({round(usdt_profit, 4)} USDT)\n"
            text += f"Profit(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(profit_slippage, 4)} WBNB ({round(usdt_profit_slippage, 4)} USDT)\n"
            text += f"Pnl: {round(pnl, 4)} WBNB ({round(usdt_pnl, 4)} USDT)\n"
            text += f"Pnl(calculate with {round(allow_slippage_percent * Decimal('100'), 1)}% slippage): {round(pnl_slippage, 4)} WBNB ({round(usdt_pnl_slippage, 4)} USDT)\n\n"
            texts.append(text)
            update_trades.append(data)

        if len(update_trades) >= num_trades:
            break

    if not test_mode and texts:
        for text in texts:
            print(f"<b>[BSC - PancakeSwap]</b>\n{text}")

    return update_trades


def main():
    start = time.time()
    all_pairs = json.load(open(pancakeswap_pairs_path))
    pairs = selectPairs(all_pairs)
    print(f"pairs: {len(pairs)}")

    try:
        pairs = get_reserves_batch_mt(pairs)
    except Exception as e:
        logger.critical('get_reserves err:', e)
        return
    end = time.time()
    print(f"update cost: {end - start} s")

    trades = findArb(pairs, tokenIn, tokenOut, maxHops, currentPairs, path, bestTrades)
    if len(trades) == 0:
        logger.info('No trades.')
        return

    trades = send_message(trades, num_trades, False)
    if len(trades) == 0:
        logger.info('No trades can be profitable.')
        return


if __name__ == "__main__":
    logger.info("START")
    main()
    logger.info("END")
