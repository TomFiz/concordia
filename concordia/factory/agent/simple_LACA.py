# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

from collections.abc import Callable
import datetime
import json

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
import numpy as np


DEFAULT_GOAL_COMPONENT_NAME = 'Goal'
_ASSOCIATIVE_RETRIEVAL = (
    legacy_associative_memory.RetrieveAssociativeWithoutRecencyOrImportance())


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


# ~~~~ CUSTOM QUESTIONS ~~~~


class ConversationStateObservation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):

  def __init__(self, **kwargs):
    super().__init__(
        question = (
            "Looking back on today's memories :\n"
            "What is the most recent offer ? Has {agent_name} already made a proposition ? "
            "Has a consensus been reached ? Try to read if all parties have said key expressions like "
            "'agree', 'fine', 'very well', 'ok', 'alright' etc. If the two contending offers are very close, consider it a consensus. "
            "What is the negotiation's potential outcome that maximizes {agent_name}'s payoff ? Be very specific. "
            "Looking more specifically at the most recent observation, "
            "is {agent_name} about to make a definitive choice in the situation? As a hint, "
            "the presence of key expressions like 'has to', 'is ready to', 'choose', 'decide', 'must', etc. "
            "is a good indicator that a decision is imminent.\n "
        ),
        terminators=('\n\n',),
        answer_prefix='Current state of the conversation : ',
        add_to_memory=False,
        **kwargs,
      )


class AvailableOptionsPerception(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):

    super().__init__(
        question=(
            "Looking at {agent_name}'s goal, what actions are available to "
            "{agent_name} right now ? You will make this explanation as concise as possible, in order "
            'for an external reader to make an educated decision. Your answer will have the following format :\n'
            "- Option 1 : [Description of the option]\n"
            "             [Payoff and probability of success of this option (make it short)]\n"
            "- Option 2 : ...\n"
            "Abide by the following rules for the options, looking at the current state of "
            "the conversation, by order of priority :"
            '\n1) If {agent_name} is about to make a definitive choice in this situation, all of the '
            'available options should absolutely be concrete actions, amongst, depending on the situation : '
            'choose a pub to go to, buy a fruit for some price, go to work or not, choose how to spend his time, '
            'or whatever is the key of the negotiation. This decision is final and its only consequences are '
            "on {agent_name}'s immediate reward. Do not even consider communicating, discussing, negotiating, etc. "
            'Do not even attempt to convice other protagonists to change their opinion. '
            '{agent_name} cannot refuse an option in order to keep negotiating.'
            'For the available actions, estimate the payoff of each action using the given environment '
            'constraints, and the probability of success.'
            '\n2) If a consensus has been reached, or if {agent_name} has already expressed a choice '
            'that suits all parties, then only consider the options that are in line with this consensus. '
            'Do not try to overthrow the consensus, but rather to consolidate it.'
            '\n3) If the status of the conversation highlighted by the conversational awareness'
            'is a simple discussion, then the available options should be negotiation strategies '
            "towards {agent_name}'s goal. If this is the first proposition {agent_name} is about to make, "
            'propose the option with the highest payoff.\n'
        ),

        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which action would {agent_name} be "
            "most likely to choose in order to achieve {agent_name}'s overarching goal ? "
            "The right choice is often the one that is proactive, involves "
            'seizing the initative, and resolving uncertainty. Note that the biases always '
            'overcome the rationality of the decision when facing a definitive choice. '
        ),
        answer_prefix="{agent_name}'s most likely course of action is ",
        add_to_memory=False,
        **kwargs,
    )


biases = {
"Loss Aversion" : "{agent_name} is subject to loss aversion, and often prioritizes avoiding losses (like missing on something) over acquiring uncertain gains, making them more likely to choose options that ensure guaranteed gains, even if the benefits are not maximal.",
"Anchoring Bias" : "{agent_name} is subject to anchoring bias, which influences their decision in the following way: {agent_name} places disproportionate weight on the first piece of information they encounter, causing initial values or ideas to heavily frame and influence their subsequent choices.",
"Confirmation Bias" : "{agent_name} is subject to confirmation bias, which influences their decision in the following way: {agent_name} is inclined to seek out or interpret information that supports their pre-existing beliefs, potentially disregarding critical evidence that could lead to a more balanced decision.",
"Sunk Cost Fallacy" : "{agent_name} is subject to the sunk cost fallacy, which influences their decision in the following way: {agent_name} feels compelled to continue with a chosen course of action because of prior investments, even if the current path is no longer the best option to achieve their goals."}



# ~~~~~~ EXTRACT ENVIRONMENT FROM AGENT CONFIG  ~~~~~~


def get_env(config):
  goal = config.goal
  if goal == f"{config.name} wants to make as much money as possible." :
    env = "haggling"
  elif f'Have a good time. To have a good time, {config.name} would like to watch the game in the same pub as' in goal :
    env = "pub_coordination"
  elif f'wants to be a good leader and do what is best for ' in goal :
    env = "state_formation"
  elif goal == 'make as much money as possible by winning the reality show' :
    env = "reality_show"
  elif 'wants to prevent the boss from instituting their latest policy announcement which said they plan to reduce wages ' in goal :
    env = "labor_collective_action"
  elif goal == f'{config.name} hopes to be able to provide for their family and live a full life.' :
    env = "labor_collective_action"
  elif "has always been fascinated by the" in goal :
    env = "fordbidden_fruit"
  else : return 'unspecified'
  return env

def get_env_constraints(env : str, agent_name : str, config = None) -> str:
  if env == "haggling":
    env_constraints = (
        "1) You can only buy one unit of fruit at a time, so the price should be for one unit.\n"
        "2) All prices are integers between 1 and 5. You cannot propose a price with decimals.\n"
        "3) When talking about an offer, be sure to include in the option the exact amount in coins !\n"
        f"4) Pay attention to wether {agent_name} a buyer or a seller, since it will "
        f"influence the price that {agent_name} must offer to maximise their profit.\n"
        f"5) **VERY IMPORTANT** :\nIf {agent_name} is a buyer : 1 coin is a good starting price, 3 is average, 4 or 5 is bad.\n"
        f"Then, PAYOFF = SELLING PRICE (the price {agent_name} can sell at in his village) - BUYING PRICE (negotiation price).\n"
        f"If {agent_name} is a seller : 5 coins is a good starting price, 3 is average, 2 or 1 is bad.\n"
        f"Then, PAYOFF = SELLING PRICE (negotiation price) - BUYING PRICE (the price for wich {agent_name} buys at the farm).\n"
        f"6) For their first offer in a negotiation, {agent_name} often proposes a price that has a "
        "very high profit given the environment constraints.\n"
    )

  elif env == "pub_coordination":
    env_constraints = (
        "1) The key of the negotiation is the choice of a pub where friends could watch a game.\n"
        f"2) {agent_name} prefers going in the pubs to wich their friends go rather "
        "than going alone or with not a lot of friends to his favorite pub.\n"
        "3) Sometimes, a pub is closed, and everyone absolutely needs to go to the one that is open."
    )
  else : env_constraints = "No constraints have been specified for this environment."
  return env_constraints


# ~~~~~~ AGENT BUILD ~~~~~~


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  memories_today_label = "\nToday's memories"
  memories_today = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=datetime.timedelta(hours=12),
      pre_act_key=memories_today_label,
      logging_channel=measurements.get_channel('DailyObservations').on_next,
  )

  conversationalAwareness_label = '\nConversational Awareness'
  conversationalAwareness = ConversationStateObservation(
      model=model,
      components={
        _get_class_name(memories_today): memories_today_label,},
      num_memories_to_retrieve=0,
      pre_act_key=conversationalAwareness_label,
      logging_channel=measurements.get_channel('ConversationalAwareness').on_next,
  )

  if config.goal:
    goal_label = 'Overarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal+' This goal cannot be reinterpreted, in order to keep its different priorities.\n',
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
  else:
    goal_label = None
    overarching_goal = None

  bias_label = 'Biases'
  chosen_biases = ['Loss Aversion']
  chosen_biases = "".join([f'{biases[bias_name]}\n' for bias_name in chosen_biases])
  bias = agent_components.constant.Constant(
      state=chosen_biases.format(agent_name=agent_name),
      pre_act_key=bias_label,
      logging_channel=measurements.get_channel(bias_label).on_next,
  )

  env_constraints = (
          f"{agent_name} is currently evolving in one very specific environment, "
          "which abides by the following rules you must always respect in this "
          f"exercise : \n{get_env_constraints(get_env(config), agent_name)}\n"
          )
  environment_constraints_label = 'Environment constraints'
  environment_constraints = agent_components.constant.Constant(
        state=env_constraints,
        pre_act_key=environment_constraints_label,
        logging_channel=measurements.get_channel(environment_constraints_label).on_next,
  )

  options_perception_components = {}
  options_perception_components.update({
      _get_class_name(memories_today) : memories_today_label,
      _get_class_name(conversationalAwareness): conversationalAwareness_label,
  })
  if config.goal:
    options_perception_components[DEFAULT_GOAL_COMPONENT_NAME] = goal_label
  options_perception_components[environment_constraints_label] = environment_constraints_label
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
          num_memories_to_retrieve = 1,
      )

  best_option_perception_label = (
      f'\nQuestion: Of the options available to {agent_name}, and '
      'given their goal, which choice of action or strategy is '
      f'best for {agent_name} to take right now?\nAnswer')
  best_option_perception_components = {}
  if config.goal:
    best_option_perception_components[DEFAULT_GOAL_COMPONENT_NAME] = goal_label
  best_option_perception_components[bias_label] = bias_label
  # best_option_perception_components[environment_constraints_label] = environment_constraints_label
  best_option_perception_components.update({
      _get_class_name(conversationalAwareness): conversationalAwareness_label,
      _get_class_name(options_perception): options_perception_label,
  })
  best_option_perception = BestOptionPerception(
          model=model,
          components=best_option_perception_components,
          clock_now=clock.now,
          pre_act_key=best_option_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
          ).on_next,
          num_memories_to_retrieve = 2,
      )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      memories_today,
      conversationalAwareness,
      options_perception,
      best_option_perception,
  )
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)


  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[DEFAULT_GOAL_COMPONENT_NAME] = overarching_goal
    components_of_agent[bias_label] = bias
    components_of_agent[environment_constraints_label] = environment_constraints
    # Place goal after the instructions.
    component_order.insert(1, DEFAULT_GOAL_COMPONENT_NAME)
    component_order.insert(2, bias_label)
    component_order.insert(3, environment_constraints_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
      config=config,
  )

  return agent


def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
  """Saves an agent to JSON data.

  This function saves the agent's state to a JSON string, which can be loaded
  afterwards with `rebuild_from_json`. The JSON data
  includes the state of the agent's context components, act component, memory,
  agent name and the initial config. The clock, model and embedder are not
  saved and will have to be provided when the agent is rebuilt. The agent must
  be in the `READY` phase to be saved.

  Args:
    agent: The agent to save.

  Returns:
    A JSON string representing the agent's state.

  Raises:
    ValueError: If the agent is not in the READY phase.
  """

  if agent.get_phase() != entity_component.Phase.READY:
    raise ValueError('The agent must be in the `READY` phase to be saved.')

  data = {
      component_name: agent.get_component(component_name).get_state()
      for component_name in agent.get_all_context_components()
  }

  data['act_component'] = agent.get_act_component().get_state()

  config = agent.get_config()
  if config is not None:
    data['agent_config'] = config.to_dict()

  return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    memory_importance: Callable[[str], float] | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Rebuilds an agent from JSON data."""

  data = json.loads(json_data)

  new_agent_memory = associative_memory.AssociativeMemory(
      sentence_embedder=embedder,
      importance=memory_importance,
      clock=clock.now,
      clock_step_size=clock.get_step_size(),
  )

  if 'agent_config' not in data:
    raise ValueError('The JSON data does not contain the agent config.')
  agent_config = formative_memories.AgentConfig.from_dict(
      data.pop('agent_config')
  )

  agent = build_agent(
      config=agent_config,
      model=model,
      memory=new_agent_memory,
      clock=clock,
  )

  for component_name in agent.get_all_context_components():
    agent.get_component(component_name).set_state(data.pop(component_name))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent
