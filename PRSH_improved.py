## INSTRUCTIONS: Drop this into BSE in place of current PRSH2 (DO NOT REPLACE PRSH, NEED FOR COMPARISON TESTS)
## Main area for change is in the respond function as this is where the hill climber is      

class PRSH2(Trader):

    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = 'PRSH: %s active_strat=[%d]:\n' % (self.tid, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string


    def __init__(self, ttype, tid, balance, time):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR

        verbose = False
        
        Trader.__init__(self, ttype, tid, balance, time)
        self.theta0 = 100         # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.k = PRSH_k                  # number of hill-climbing points (cf number of arms on a multi-armed-bandit)
        self.strat_wait_time = 60  # how many secs do we give any one strat before switching? todo: make this randomized withn some range
        self.strat_range_min = 0.75 # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = 0.75 # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None        # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time   # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.01 * random.random()    # minimum profit-per-sec difference between strategies that counts
        self.strats=[]              # strategies awaiting initialization
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max
        self.mutate_strat = PRSH_mutate_strat  # how to mutate the strategy values when hill-climbing

        for s in range(0, self.k):
            # initialise each of the strategies in sequence
            start_time = time
            profit = 0.0
            profit_per_second = 0
            lut_bid = None
            lut_ask = None
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                strategy = self.mutate_strat(self.strats[0]['stratval'], self.k)     # mutant of strats[0]
            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})

        if verbose:
            print("PRSH %s %s\n" % (tid, self.strat_str()))


    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1   # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1   # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # dirn is direction: -1 for buy, +1 for sell

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.000001 #used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: PRZI.getorder() self.strat out of range\n')

            if (dirn != 1.0) and (dirn != -1.0):
                # out of range
                sys.exit('PRSH FAIL: PRZI.calc_cdf() bad dirn\n')

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax < pmin\n')

            dxs = dirn * strat

            if verbose:
                print('PRSH calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf=[{'price':pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0   # just in case
                calp_interval.append({'price':p, "cal_p":cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p-pmin]['price'] # todo: what does this do?
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob}) #todo shouldnt ths be "price" not "p"?

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

        verbose = False

        if verbose:
            print('t=%f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                dummy = 1

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']         # so use that as my new estimate of highest
                    self.pmax = maxprice

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # it may be more efficient to detect the ZIC special case and generate a price directly
            # whether it is or not depends on how many entries need to be sampled in the LUT reverse-lookup
            # versus the compute time of the call to random.randint that would be used in direct ZIC
            # here, for simplicity, we're not treating ZIC as a special case...
            # ... so the full CDF LUT needs to be instantiated for ZIC (strat=0.0) just like any other strat value

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            if otype == 'Bid':

                # direction * strat
                dxs = -1 * strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or\
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, -1, p_min, p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:   # otype == 'Ask'

                dxs = strat

                p_min = int(limit)
                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, +1, p_min, p_max)

                lut = self.strats[self.active_strat]['lut_ask']

            if verbose:
                # print('PRZI LUT =', lut)
                # print ('[LUT print suppressed]')
                dummy = 1

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        # Trader.bookkeep(self, trade, order, verbose, time) -- todo: calls all of the above?

        # todo: expand from here

        # Check: bookkeep is only called after a successful trade? i.e. no need to check re trade or not

        # so ...
        # if I have just traded and I am a PRSH trader
        # then I want to reset the timer on the current strat and update its profit sum

        self.strats[self.active_strat]['profit'] += profit


    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self, time, lob, trade, verbose):

        #shc_algo = 'basic'

         # "basic" is a very basic form of stochastic hill-cliber (SHC) that v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other strats
        # then all counters are reset, and this is repeated indefinitely
        # todo: add in other shc_algo that are cleverer than this,
        # e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        shc_algo = 'softmax'

        verbose = False

        # first update each strategy's profit-per-second value -- this is the "fitness" of each strategy
        for s in self.strats:
            pps_time = time - s['start_t']
            if pps_time > 0:
                s['pps'] = s['profit'] / pps_time
            else:
                s['pps'] = 0.0

##########################################################################################################
        if shc_algo == 'epsilon_greedy': # reference book       
            dummy  = 1 # placeholder
        # add upper confidence bound/other different MAB solvers?
        elif shc_algo == 'softmax':

            def softmax(z):
                output = []
                for i in z:
                    output.append(np.exp(i)/np.sum(z))
                return output
            
            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats
            for s in self.strats:
                # assume that all strats have had long enough, and search for evidence to the contrary
                all_old_enough = True
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough - which to choose:
                # apply softmax to profits

                self.strats = softmax(self.strats)

                # sort by softmax
                strats_sorted = sorted(self.strats, key = lambda k: k['pps'], reverse = True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                # if the difference between the top two strats is too close to call then flip a coin
                # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                prof_diff = strats_sorted[0]['profit'] - strats_sorted[1]['profit']
                if abs(prof_diff) < self.profit_epsilon / 100:          # divide by 100 since softmax reduces range to 0, 1
                    # they're too close to call, so just flip a coin
                    best_strat = random.randint(0, 1)
                elif prof_diff > 0:
                    best_strat = 0
                else:
                    best_strat = 1

                if best_strat == 1:
                    # need to swap strats[0] and strats[1]
                    tmp_strat = strats_sorted[0]
                    strats_sorted[0] = strats_sorted[1]
                    strats_sorted[1] = tmp_strat

                # the sorted list of strats replaces the existing list
                self.strats = strats_sorted

                # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate
                # record it
                
                # now replicate and mutate elite into all the other strats
                for s in range(1, self.k):    # note range index starts at one not zero
                    self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], self.k)
                    self.strats[s]['start_t'] = time
                    self.strats[s]['profit'] = 0.0
                    self.strats[s]['pps'] = 0.0
                # and then update (wipe) records for the elite
                self.strats[0]['start_t'] = time
                self.strats[0]['profit'] = 0.0
                self.strats[0]['pps'] = 0.0   
##############################################################################################################
        if shc_algo == 'basic':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                dummy = 1

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.
            # when a trader does a deal, clock is reset; todo check this!!!
            # clock also reset when new a strat is created, obvs. todo check this!!! also check bookkeeping/proft etc

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            for s in self.strats:
                # assume that all strats have had long enough, and search for evidence to the contrary
                all_old_enough = True
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key = lambda k: k['pps'], reverse = True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

                # if the difference between the top two strats is too close to call then flip a coin
                # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                prof_diff = strats_sorted[0]['profit'] - strats_sorted[1]['profit']
                if abs(prof_diff) < self.profit_epsilon:
                    # they're too close to call, so just flip a coin
                    best_strat = random.randint(0,1)
                elif prof_diff > 0:
                    best_strat = 0
                else:
                    best_strat = 1

                if best_strat == 1:
                    # need to swap strats[0] and strats[1]
                    tmp_strat = strats_sorted[0]
                    strats_sorted[0] = strats_sorted[1]
                    strats_sorted[1] = tmp_strat

                # the sorted list of strats replaces the existing list
                self.strats = strats_sorted

                # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate
                # record it
                
                # now replicate and mutate elite into all the other strats
                for s in range(1, self.k):    # note range index starts at one not zero
                    self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], self.k)
                    self.strats[s]['start_t'] = time
                    self.strats[s]['profit'] = 0.0
                    self.strats[s]['pps'] = 0.0
                # and then update (wipe) records for the elite
                self.strats[0]['start_t'] = time
                self.strats[0]['profit'] = 0.0
                self.strats[0]['pps'] = 0.0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

        else:
            sys.exit('FAIL: bad value for shc_algo')
    
     