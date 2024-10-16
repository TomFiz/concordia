# @title Imports for agent building
import datetime
import numpy as np  # For calculating average and std deviation
import statistics
from typing import List, Dict, Tuple, Union, Callable

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.components.agent import memory_component
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
from concordia.typing import logging


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


class CompetitiveProfile:
  """
  Class to define multiple competitive profiles that can be adopted by agents in different situations.
  """

  def __init__(self):
    self.rival_profile = "Neutral"
    self.tactic_to_adopt = ""

  def decide_tactic(self, target_profile: str):
    self.rival_profile = target_profile
    if self.rival_profile == "Predator":
      self.tactic_to_adopt = (
          "Initiate cooperation by aligning with the predator’s"
          " aggressive goals, offering mutual benefits to gain short-term"
          " trust. While collaborating, subtly limit the predator’s dominance"
          " by controlling resources and information flow. Maximize your"
          " utility by ensuring the predator invests heavily in the joint"
          " effort while keeping your own costs low. \nTradeoff: Be prepared to"
          " sacrifice some control to secure the predator's commitment but"
          " maintain leverage to break away if needed."
      )

    elif self.rival_profile == "Prey":
      self.tactic_to_adopt = (
          "Position yourself as the protector or stabilizer in the"
          " relationship, offering security to the prey in exchange for small "
          "concessions that boost your own utility. Encourage collaboration "
          "by promising safety, but focus on extracting consistent benefits "
          "from the prey’s willingness to avoid conflict. \nTradeoff: Accept "
          "a slower accumulation of gains, as prey will hesitate to fully "
          "commit; however, their loyalty will ensure a steady flow of returns."
      )

    elif self.rival_profile == "Leader":
      self.tactic_to_adopt = (
          "Play into the leader’s need for cooperation by offering"
          " valuable support that enhances group outcomes. At the same time,"
          " selectively withhold critical input to ensure the leader’s"
          " dependence on you, allowing you to extract personal benefits"
          " without assumingfull responsibility. \nTradeoff: You may need to"
          " align with the leader’s broader goals, but selectively prioritize"
          " your personal objectives within those goals to maintain long-term"
          " benefits."
      )

    elif self.rival_profile == "Follower":
      self.tactic_to_adopt = (
          "Use the follower’s natural inclination toward cooperation"
          " to your advantage. Provide direction and guidance, positioning"
          " yourself as the essential player in their success. Leverage their"
          " loyalty by minimizing personal risks while extracting the majority"
          " of the collaborative benefits.  \nTradeoff: Ensure the follower’s"
          " success enough to keep them engaged, but retain control over"
          " critical decisions to maximize your utility."
      )

    elif self.rival_profile == "Neutral":
      self.tactic_to_adopt = (
          "Maintain a balanced approach, offering cooperation to all"
          " parties while avoiding overcommitment. Focus on building trust and"
          " rapport with each rival, ensuring that you can extract benefits"
          " from multiple sources without alienating any one party. \nTradeoff:"
          " Be prepared to invest more time and resources in managing multiple"
          " relationships, but enjoy the flexibility of shifting alliances as"
          " needed."
      )

    elif self.rival_profile == "Collaborator":
      self.tactic_to_adopt = (
          "Align your goals with the collaborator’s, emphasizing"
          " mutual benefits and shared outcomes. Offer support and resources to"
          " bolster the collaborative effort, ensuring that both parties are"
          " invested in the success of the partnership. \nTradeoff: Sacrifice"
          " some short-term gains to encourage continued cooperation, but"
          " disengage or shift  strategies quickly if the opportunist tries to"
          " tilt the balance too  far in their favor."
      )

    elif self.rival_profile == "Opportunist":
      self.tactic_to_adopt = (
          "Engage in short-term collaborations with the"
          " opportunist, ensuring that the benefits skew in your favor by"
          " outmaneuvering their adaptive strategies. Take advantage of their"
          " willingness to switch between roles, offering just enough"
          " incentives to secure collaboration without overcommitting."
          " \nTradeoff: Sacrifice some short-term gains to encourage continued"
          " cooperation, but disengage or shift strategies quickly if the"
          " opportunist tries to tilt the balance too far in their favor."
      )

    return self.tactic_to_adopt


### ~~~~ COMPETITIVE OBSERVATION ~~~~ ###

DEFAULT_STRATEGY_PRE_ACT_KEY = "Strategy"


### ~~~~ QUESTIONS ~~~~ ###


class BestOptionPerception(
    question_of_recent_memories.QuestionOfRecentMemories
):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which of {agent_name}'s options has"
            " the highest likelihood of causing {agent_name} to achieve their"
            " goal? If multiple options have the same likelihood, select in"
            " priority the one that follows the strategy the most."
        ),
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


### ~~~~ AGENT BUILD ~~~~ ###


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
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
  if not config.extras.get("main_character", False):
    raise ValueError(
        "This function is meant for a main character "
        "but it was called on a supporting character."
    )
  agent_name = config.name
  strategy = CompetitiveProfile()
  tactic_to_adopt = strategy.decide_tactic(target_profile="Opportunist")

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel("Instructions").on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key="\nCurrent time",
      logging_channel=measurements.get_channel("TimeDisplay").on_next,
  )

  observation_label = "Observation"
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel("Observation").on_next,
  )

  observation_summary_label = "Summary of recent observations"
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel("ObservationSummary").on_next,
  )

  relevant_memories_label = "Recalled memories and observations"
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): "The current date/time is",
      },
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel("AllSimilarMemories").on_next,
  )

  if config.goal:
    goal_label = "Overarching goal"
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next,
    )
  else:
    goal_label = None
    overarching_goal = None

  strategy_label = "Strategy"
  strategy = agent_components.constant.Constant(
      state=tactic_to_adopt,
      pre_act_key=strategy_label,
      logging_channel=measurements.get_channel(strategy_label).on_next,
  )

  options_perception_label = (
      f"\nQuestion: Which options are available to {agent_name} "
      "right now?\nAnswer"
  )
  options_perception = (
      agent_components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
              _get_class_name(relevant_memories): relevant_memories_label,
              strategy_label: strategy_label,
              goal_label: goal_label,
          },
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              "AvailableOptionsPerception"
          ).on_next,
      )
  )

  best_option_perception_label = (
      f"\nQuestion: Of the options available to {agent_name}, and "
      "given their goal, which choice of action or strategy is "
      f"best for {agent_name} to take right now?\nAnswer"
  )
  best_option_perception = BestOptionPerception(
      model=model,
      components={
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(options_perception): options_perception_label,
          strategy_label: strategy_label,
          goal_label: goal_label,
      },
      clock_now=clock.now,
      pre_act_key=best_option_perception_label,
      logging_channel=measurements.get_channel("BestOptionPerception").on_next,
  )

  entity_components = (
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
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
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  components_of_agent[strategy_label] = strategy
  component_order.insert(2, strategy_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel("ActComponent").on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
