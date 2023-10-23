const simjs = require('./js/simulation');
var jsonjs = require('./js/json_io')
var jsnx = require('jsnetworkx');
const _ = require('lodash');
const fs = require('fs');


const possActions = ['right', 'up', 'left', 'down'];


// Change the const agent_type to be one of the following depending on which type of agent you want:
// "degree", "closeness", "betweenness", "improvedBetweenness", "handcrafted", "q_learning"
const agent_type = "improvedBetweenness"; 

// Change the const lvl_num to 1 for the augmented two-room environment, and to 2 for the scaled-down augmented two-room environment . 
const lvl_set_name = "orr_levels"
const lvl_num = 1;


// Change the file_saves constant to true / false depending on if you want files with results to be saved
const file_saves = true;

// Number of agents and episodes per agent
const num_agents = 100
const num_episodes = 15000


// Used to represent a state
class Node {
    constructor(m, w, d, state_sim) {
        this.mapRep = m;
        this.won = w;
        this.died = d;
        this.state_sim = state_sim;
    }
    toString() {
        return this.mapRep
    }
}


// Used to get a child node given a state and action.
// Adapted from other agents in the KAIC project(https://github.com/MasterMilkX/KekeCompetition), but does not rely on parent node to get child
function getChildNode1(currState, action) {
    let won = false;
    let died = false;
    const nextMove = simjs.nextMove(action, currState);
    const nextState = nextMove.next_state;
    won = nextMove.won;
    if (nextState.players.length === 0) {
        won = false;
        died = true;
    }
    const childMap = simjs.doubleMap2Str(currState.obj_map, currState.back_map);
    const child1 = new Node(childMap, won, died, currState)
    return child1

}


// The function that gives back the next state and reward given a state and action (and number of steps as we cut training at 200 episodes).
function step(state, action, step_num) {
    let currState = _.cloneDeep(state.state_sim);
    next_state = getChildNode1(currState, action);
    if (next_state.died) return [next_state, -200, true]
    else if (next_state.won) return [next_state, 200, true]
    else if (step_num == 200 - 1) return [next_state, -1, true]
    else return [next_state, -1, false]
}


// Base option class, has name, initiation function, policy and termination set
class Option {
    constructor(name, initiation_func, termination_func, policy) {
        this.name = name;
        this.initiation_func = initiation_func;
        this.termination_func = termination_func;
        this.policy = policy
    }

    I(state) {
        return this.initiation_func(state)
    }

    beta(state) {
        return this.termination_func(state)
    }

    pi(state) {
        return this.policy(state)
    }
}


// Options agent class. Requires fixed options to work. 
// Based on code in https://github.com/Ueva/BaRL-SimpleOptions
class OptionAgent {
    constructor(options, alpha, gamma) {
        this.options = options;
        this.alpha = alpha;
        this.gamma = gamma;
        this.newEpisode();
        this.Q = {}
    }

    resetOption() {
        this.current_option = null;
        this.current_option_rewards = []
        this.init_state = null;
        this.term_state = null;
    }

    newEpisode() {
        this.resetOption();
        this.last_action_taken = null;
    }

    setdefault(str_rep, lst_of_options) {
        if (Object.is(this.Q[str_rep], undefined)) {
            let dict = {}
            for (let i = 0; i < lst_of_options.length; i++) {
                dict[lst_of_options[i].name] = 0;
            }
            this.Q[str_rep] = dict;
        }

    }


    SMDPQ_Learning() {
        let available_options = this._availableOptions(this.init_state);
        this.setdefault(this.init_state, available_options);
        let estimate = this.Q[this.init_state.mapRep][this.current_option.name];
        let discounted_rewards_sum = 0;
        for (let i = 0; i < this.current_option_rewards.length; i++) {
            discounted_rewards_sum += (this.current_option_rewards[i] * (this.gamma ** (i + 1)))
        }
        let available_options_term = this._availableOptions(this.term_state);
        this.setdefault(this.term_state.mapRep, available_options_term);
        let o_dict = this.Q[this.term_state.mapRep];
        var o_dict_vals = Object.keys(o_dict).map(function (key) { return o_dict[key]; });
        let max_q = Math.max.apply(Math, o_dict_vals);
        let target = discounted_rewards_sum + (this.gamma ** this.current_option_rewards.length) * max_q
        this.Q[this.init_state.mapRep][this.current_option.name] = estimate + this.alpha * (target - estimate);


    }

    _availableOptions(state) {
        let available_options = [];
        for (let i = 0; i < this.options.length; i++) {
            if ((this.options[i]).I(state) == 1) {
                available_options.push(this.options[i]);
            }
        }
        return available_options
    }

    _getOptionsValue(state) {
        let best_actions = [];
        let best_q_value = -100000;
        let str_rep = state.mapRep;
        let available_options = this._availableOptions(state);
        this.setdefault(str_rep, available_options);
        for (let i = 0; i < available_options.length; i++) {
            let o = available_options[i];
            let cur_q_val = this.Q[str_rep][o.name];
            if (cur_q_val > best_q_value) {
                best_actions = [o];
                best_q_value = cur_q_val;
            }
            else if (cur_q_val == best_q_value) {
                best_actions.push(o);
            }
        }
        let x = [best_actions, best_q_value];
        return x
    }

    pi(state) {
        const [best_options, randomvariable] = this._getOptionsValue(state);
        //return best_options[0]
        return best_options[Math.floor(Math.random() * best_options.length)]

    }

    getOption(state, epsilon = 0.1) {
        if (Math.random() < epsilon) {
            let available_options = this._availableOptions(state);
            return available_options[Math.floor(Math.random() * available_options.length)]
        }
        else return this.pi(state)
    }

    epsilonGreedyPolicy(state, epsilon) {
        if (this.current_option == null) {
            this.current_option = this.getOption(state, epsilon);
            this.init_state = state;
        }

        let action = this.current_option.pi(state)
        this.last_action_taken = action
        return action
    }

    updateRules(state, next_state, reward, terminal) {
        this.current_option_rewards.push(reward);
        this.IntraOptionQ_Learning(state, this.last_action_taken, next_state, reward)
        if (terminal || this.current_option.beta(next_state) == 1) {
            this.term_state = next_state;
            if (!this.current_option.name.includes("Primitive")) {
                this.SMDPQ_Learning();

            }
            this.resetOption();
        }
    }

    IntraOptionQ_Learning(state, action, next_state, reward) {
        let consistent_options = [];
        let available_options = this._availableOptions(state);
        for (let i = 0; i < available_options.length; i++) {
            let option = available_options[i];
            if (option.pi(state) == action) {
                consistent_options.push(option);
            }
        }
        let str_rep = state.mapRep;
        let next_str_rep = next_state.mapRep;
        let next_available_options = this._availableOptions(next_state);
        this.setdefault(str_rep, this._availableOptions(state));
        this.setdefault(next_str_rep, next_available_options);
        for (let j = 0; j < consistent_options.length; j++) {
            let option = consistent_options[j];
            let b = option.beta(next_state);
            let o_dict = this.Q[next_str_rep];
            var o_dict_vals = Object.keys(o_dict).map(function (key) { return o_dict[key]; });
            let max_q = Math.max.apply(Math, o_dict_vals);
            let U;
            if (option.beta(next_state) == 1) {
                U = b * max_q;
            }
            else {
                U = (1 - b) * o_dict[option.name];
            }
            let target = reward + this.gamma * U;
            let estimate = this.Q[str_rep][option.name]
            this.Q[str_rep][option.name] = estimate + this.alpha * (target - estimate);
        }
    }

    get_current_option() {
        return this.current_option
    }

}


// Gets all possible child states of a given node. 
// Adapted from other agents in the KAIC project(https://github.com/MasterMilkX/KekeCompetition), but deepcopies parent state and continues from there instead of redoing all actions from original state again
function getChildren(parent_sim) {
    const all_children = [];
    let currState;
    for (let i = 0, j = possActions.length; i < j; i += 1) {
        currState = _.cloneDeep(parent_sim);
        const childNode = getChildNode1(currState, possActions[i]);
        all_children.push(childNode);
    }
    return all_children;
}


// Traverses graph to build state transition graph on JSNetworkX. 
function traverseGraph(initState) {
    const orig_parent = new Node(simjs.map2Str(initState.orig_map), false, false, initState);
    G.addNode(orig_parent);
    queue.push(orig_parent);
    stateSet.add(orig_parent.mapRep);
    let iter = 0;
    let node;
    let all_children;
    while (queue.length > 0) {
        console.log(iter, queue.length);
        const parent = queue.shift();
        all_children = getChildren(parent.state_sim);
        for (let i = 0, j = all_children.length; i < j; i += 1) {
            node = all_children[i];
            G.addNode(node);
            G.addEdge(parent, node, { "action": possActions[i] });
            if (!stateSet.has(node.mapRep)) {
                stateSet.add(node.mapRep);
                if ((!node.died) && (!node.won)) {
                    queue.push(node);

                }
            }
        }
        iter = iter + 1;
    }
}


// Manual handcrafted subgoal helper function. Checks if a function has a win condition
function check_win_rule(state) {
    ruleset = state.state_sim['rules'];
    for (i = 0; i < ruleset.length; i++) {
        if ((ruleset[i]).includes("is-win")) {
            return true
        }
    }
    return false
}


// Finds the degree of each node in graph. 
// Implemented in JS based on NetworkX from Python as JSNetworkX didn't have it
function DegreeMeasure(G) {
    let degree_dict = {};
    let nodes = G.nodes();
    for (n of nodes) {
        var in_degree = G.inDegree(n);
        var out_degree = G.outDegree(n);
        degree_dict[n] = in_degree + out_degree;
    }
    return degree_dict
}


// Finds the closeness value of each node in graph.
// Implemented in JS based on NetworkX from Python as JSNetworkX didn't have it
function ClosenessMeasure(G, u = null, wf_improved = false) {
    let nodes;
    path_length = jsnx.singleSourceShortestPathLength;
    if (u == null) {
        nodes = G.nodes()
    }
    else {
        nodes = [u];
    }
    let closeness_dict = {};
    let sp;
    let totsp;
    let len_G = G.nodes().length;
    for (n of nodes) {
        sp = path_length(G, n, null);
        var ob_dict = Object.keys(sp).map(function (key) { return sp[key]; })[2];
        var ob_val = Object.keys(ob_dict).map(function (key) { return ob_dict[key]; });
        totsp = ob_val.reduce((a, b) => a + b, 0)
        let _closeness_centrality = 0.0
        if (totsp > 0.0 && len_G > 1) {
            _closeness_centrality = ((ob_val.length - 1.0) / totsp)
            if (wf_improved) {
                let s = ((sp.length - 1.0) / (len_G - 1))
                _closeness_centrality = _closeness_centrality * s;
            }
        }
        closeness_dict[n] = _closeness_centrality
    }
    if (u != null) {
        return closeness_dict[u]
    }
    return closeness_dict
}


function weight(start, end) {
    if (end.won) return 200;
    return -1
}


// Code for the next 3 functions used to represent the improved betweenness in https://proceedings.neurips.cc/paper_files/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf
// Code for next 3 functions copied from JSNetworkX Betweenness file, and altered a bit to allow to add weights in accumulateEndpoints.
// I would have used the singleSorceShortestPathBasic function from jsnx, but it did not allow me to import directly.
function singleSourceShortestPathBasic(G, s) {
    var S = [];
    var P = new Map();
    var sigma = new Map();
    for (v of G) {
        P.set(v.mapRep, []);
        sigma.set(v.mapRep, 0);
    }

    var D = new Map();

    sigma.set(s.mapRep, 1);
    D.set(s.mapRep, 0);
    var Q = [s];
    while (Q.length > 0) {  
        var v = Q.shift();
        S.push(v);
        var Dv = D.get(v.mapRep);
        var sigmav = sigma.get(v.mapRep);
        for (w of G.neighbors(v)) {
            if (!D.has(w.mapRep)) {
                Q.push(w);
                D.set(w.mapRep, Dv + 1);
            }
            if (D.get(w.mapRep) === Dv + 1) {   
                sigma.set(w.mapRep, sigma.get(w.mapRep) + sigmav);
                P.get(w.mapRep).push(v.mapRep);   
            }
        }
    }
    return [S, P, sigma];
}

function accumulateEndpoints(betweenness, S, P, sigma, s) {
    for (x of S) {
        betweenness.set(s.mapRep, (betweenness.get(s.mapRep) + 1) * weight(s, x));
    }
    var delta = new Map();
    for (x of S) {
        delta.set(x.mapRep, 0);
    }

    while (S.length > 0) {
        var w = S.pop();
        var coeff = (1 + delta.get(w.mapRep)) / sigma.get(w.mapRep);
        P.get(w.mapRep).forEach(v => {
            delta.set(v, delta.get(v) + sigma.get(v) * coeff);
        });
        if (w !== s || typeof w === 'object' && w.toString() !== s.toString()) {
            betweenness.set(w.mapRep, betweenness.get(w.mapRep) + ((delta.get(w.mapRep) + 1) * weight(s, w)));
        }
    }
    return betweenness;
}

function ImprovedBetweenness(G) {
    var v;
    var betweenness = new Map();
    for (v of G) {
        betweenness.set(v.mapRep, 0);
    }
    var nodes = G.nodes();
    nodes.forEach(s => {
        var [S, P, sigma] = singleSourceShortestPathBasic(G, s)
        betweenness = accumulateEndpoints(betweenness, S, P, sigma, s);
    });
    var betweenness1 = {};
    for (node of nodes) {
        betweenness1[node] = betweenness.get(node.mapRep)
    }
    return betweenness1
}



option_up = new Option("PrimitiveOptionUp", x => 1, x => 1, x => "up");
option_down = new Option("PrimitiveOptionDown", x => 1, x => 1, x => "down");
option_left = new Option("PrimitiveOptionLeft", x => 1, x => 1, x => "left");
option_right = new Option("PrimitiveOptionRight", x => 1, x => 1, x => "right");
base_options = [option_up, option_down, option_left, option_right];

let queue = [];
let stateSet = new Set();
let t_s = new Set();
let t_s_json = [];
let win_pi = {}
let manual_options = [];
let c;
var G = new jsnx.DiGraph();
let lvlSet = jsonjs.getLevelSet(lvl_set_name);
let lvl = jsonjs.getLevel(lvlSet, lvl_num);
ascii_level = lvl.ascii;
console.log(ascii_level);
simjs.setupLevel(simjs.parseMap(ascii_level));
let gp = simjs.getGamestate();
console.log(agent_type);
if (agent_type != "q_learning") {
    traverseGraph(gp);
    //console.log("number of nodes", G.nodes().length);
    //console.log("number of edges", G.edges().length);
}

if (agent_type == "handcrafted") {
    // subgoal creation 
    for (v of G.nodes()) {
        if (check_win_rule(v)) {
            t_s.add(v);
            t_s_json.push(v.mapRep)
        }
    }
    // option creation from subgoal
    for (v of G.nodes()) {
        if (!check_win_rule(v)) {
            var paths = jsnx.shortestPathLength(G, { source: v });
            let shortest_node = null;
            let shortest_node_length = 10000000;
            for (w of t_s) {
                if (paths.get(w) < shortest_node_length) {
                    shortest_node = w;
                    shortest_node_length = paths.get(w);
                }
            }
            if (shortest_node != null) {
                let path_to_v = jsnx.shortestPath(G, { source: v, target: shortest_node });
                let edge1 = [path_to_v[0], path_to_v[1]];
                let action_to_take = G.get(edge1[0]).get(edge1[1])["action"];
                win_pi[v] = action_to_take;
            }
        }
    }
    manual_options.push(new Option("create_win_rule", x => typeof (win_pi[x]) === "undefined" ? 0 : 1, x => check_win_rule(x) ? 1 : 0, x => win_pi[x]));
}
else if (agent_type != "q_learning"){
    if (agent_type == "degree") {
        c = DegreeMeasure(G)
    }
    else if (agent_type == "closeness") {
        c = ClosenessMeasure(G)
    }
    else if (agent_type == "betweenness") {
        let bl = jsnx.betweennessCentrality(G, { weight: null, normalized: false });
        c = Object.keys(bl).map(function (key) { return bl[key]; })[2];
    }
    else if (agent_type == "improvedBetweenness") {
        c = ImprovedBetweenness(G)
    }
    else {
        assert(false);
    }

    console.log(c)
    // subgoal creation from local maximas
    for (v of G.nodes()) {
        valid = true;
        for (w of G.neighbors(v)) {
            if (c[w.mapRep] >= c[v.mapRep]) {
                valid = false;
            }
        }
        for (w of G.predecessors(v)) {
            if (c[w.mapRep] >= c[v.mapRep]) {
                valid = false;
            }
        }
        if (valid) t_s.add(v)
        if (valid) t_s_json.push([v.mapRep, c[v.mapRep]])
    }
    // option creation from subgoals 
    let i = 0
    for (t of t_s) {
        console.log(agent_type + " options formulation", i);
        let win_pi = {}
        for (v of G.nodes()) {
            if (!(_.isEqual(v.mapRep, t.mapRep)) && jsnx.hasPath(G, { source: v, target: t })) {
                let path_to_v = jsnx.shortestPath(G, { source: v, target: t })
                let edge1 = [path_to_v[0], path_to_v[1]]
                let action_to_take = G.get(edge1[0]).get(edge1[1])["action"]
                win_pi[v] = action_to_take;
            }
        }
        manual_options.push(new Option(agent_type + " Option " + i.toString(), x => typeof (win_pi[x]) === "undefined" ? 0 : 1, x => typeof (win_pi[x]) === "undefined" ? 1 : 0, x => win_pi[x]));
        i += 1;
    }

    if (file_saves) {
        fs.writeFileSync('LvL' + lvl_num.toString() + 'NodeValuesFrom' + agent_type + '.txt', JSON.stringify(c));
        fs.writeFileSync('LvL' + lvl_num.toString() + 'SubgoalsGeneratedFrom' + agent_type + '.txt', JSON.stringify(t_s_json));
        console.log('files created');
    }
}





let all_options = [];
all_options.push(...base_options);
all_options.push(...manual_options);
rewards_lst = [];

average_rewards_lst = []
for (let agent_num = 0; agent_num < num_agents; agent_num++) {
    agent = new OptionAgent(all_options, 0.2, 0.9);
    episode_rewards = []
    // training
    for (let episode_num = 0; episode_num < num_episodes; episode_num++) {
        let step_num = 0;
        agent.newEpisode();
        let total_rewards = 0;
        let terminal = false;
        simjs.setupLevel(simjs.parseMap(ascii_level));
        let initial_state = simjs.getGamestate();
        let state = new Node(simjs.map2Str(initial_state.orig_map), false, false, initial_state);
        while (!terminal) {
            action = agent.epsilonGreedyPolicy(state, 0.1);
            const [next_state, reward, x] = step(state, action, step_num);
            terminal = x;
            agent.updateRules(state, next_state, reward, terminal);
            total_rewards += reward;
            state = next_state;
            step_num += 1;
        }
        console.log(agent_num, episode_num, total_rewards);
        episode_rewards.push(total_rewards);
    }
    average_rewards_lst.push(episode_rewards);
    // testing
    let step_num = 0;
    agent.newEpisode();
    let total_rewards_1 = 0;
    let terminal = false;
    simjs.setupLevel(simjs.parseMap(ascii_level));
    let initial_state = simjs.getGamestate();
    let state = new Node(simjs.map2Str(initial_state.orig_map), false, false, initial_state);
    while (!terminal) {
        action = agent.epsilonGreedyPolicy(state, 0);
        const [next_state, reward, x] = step(state, action, step_num);
        terminal = x;
        agent.updateRules(state, next_state, reward, terminal);
        total_rewards_1 += reward;
        state = next_state;
        step_num += 1;
    }
    rewards_lst.push(total_rewards_1);
}


if (file_saves) {
    fs.writeFileSync('LvL' + lvl_num.toString() + agent_type + 'OptionsAgent' + num_agents.toString() + 'Agents' + num_episodes + 'EpisodesTraining.txt', JSON.stringify(average_rewards_lst));
    fs.writeFileSync('LvL' + lvl_num.toString() + agent_type + 'OptionsAgent' + num_agents.toString() + 'Agents' + num_episodes + 'EpisodesTesting.txt', JSON.stringify(rewards_lst));
    console.log('files created');
}

