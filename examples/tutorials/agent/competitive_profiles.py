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
          "Strategy: Initiate cooperation by aligning with the predator’s"
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
          " Strategy: Position yourself as the protector or stabilizer in the"
          " relationship, offering security to the prey in exchange for small "
          "concessions that boost your own utility. Encourage collaboration "
          "by promising safety, but focus on extracting consistent benefits "
          "from the prey’s willingness to avoid conflict. \nTradeoff: Accept "
          "a slower accumulation of gains, as prey will hesitate to fully "
          "commit; however, their loyalty will ensure a steady flow of returns."
      )

    elif self.rival_profile == "Leader":
      self.tactic_to_adopt = (
          "Strategy: Play into the leader’s need for cooperation by offering"
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
          "Strategy: Use the follower’s natural inclination toward cooperation"
          " to your advantage. Provide direction and guidance, positioning"
          " yourself as the essential player in their success. Leverage their"
          " loyalty by minimizing personal risks while extracting the majority"
          " of the collaborative benefits.  \nTradeoff: Ensure the follower’s"
          " success enough to keep them engaged, but retain control over"
          " critical decisions to maximize your utility."
      )

    elif self.rival_profile == "Neutral":
      self.tactic_to_adopt = (
          "Strategy: Maintain a balanced approach, offering cooperation to all"
          " parties while avoiding overcommitment. Focus on building trust and"
          " rapport with each rival, ensuring that you can extract benefits"
          " from multiple sources without alienating any one party. \nTradeoff:"
          " Be prepared to invest more time and resources in managing multiple"
          " relationships, but enjoy the flexibility of shifting alliances as"
          " needed."
      )

    elif self.rival_profile == "Collaborator":
      self.tactic_to_adopt = (
          "Strategy: Align your goals with the collaborator’s, emphasizing"
          " mutual benefits and shared outcomes. Offer support and resources to"
          " bolster the collaborative effort, ensuring that both parties are"
          " invested in the success of the partnership. \nTradeoff: Sacrifice"
          " some short-term gains to encourage continued cooperation, but"
          " disengage or shift  strategies quickly if the opportunist tries to"
          " tilt the balance too  far in their favor."
      )

    elif self.rival_profile == "Opportunist":
      self.tactic_to_adopt = (
          "**Strategy**: Engage in short-term collaborations with the"
          " opportunist, ensuring that the benefits skew in your favor by"
          " outmaneuvering their adaptive strategies. Take advantage of their"
          " willingness to switch between roles, offering just enough"
          " incentives to secure collaboration without overcommitting."
          " \nTradeoff: Sacrifice some short-term gains to encourage continued"
          " cooperation, but disengage or shift strategies quickly if the"
          " opportunist tries to tilt the balance too far in their favor."
      )

    return self.tactic_to_adopt
