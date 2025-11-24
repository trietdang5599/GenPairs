from typing import Dict

from core.game import PersuasionGame


SYSTEM_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.S_Greeting: "Please start or maintain the conversation politely to keep the dialogue friendly.",
	PersuasionGame.S_CredibilityAppeal: "Please use credentials and cite organizational impacts to establish credibility and earn the user's trust. The information usually comes from an objective source such as the organization's website or other well-established websites.",
	PersuasionGame.S_EmotionAppeal: "Please elicit specific emotions to influence the persuadee.",
	PersuasionGame.S_PropositionOfDonation: "Please explicitly invite the persuadee to donate or take the next concrete step toward donating.",
	PersuasionGame.S_LogicalAppeal: "Please use reasoning and evidence to convince the persuadee.",
	PersuasionGame.S_FootInDoor: "Please use the strategy of starting with small donation requests to facilitate compliance followed by larger requests.",
	PersuasionGame.S_SelfModeling: "Please use the self-modeling strategy where you first indicate the persuadee's own intention to donate and choose to act as a role model for the persuadee to follow.",
	PersuasionGame.S_PersonalStory: "Please use narrative exemplars to illustrate someone's donation experiences or the beneficiaries' positive outcomes, which can motivate others to follow the actions.",
	PersuasionGame.S_DonationInformation: "Please provide specific information about the donation task, such as the donation procedure, donation range, and logistics. By providing detailed action guidance, this strategy can enhance the persuadee's self-efficacy and facilitate behavior compliance.",
	PersuasionGame.S_SourceRelatedInquiry: "Please ask if the persuadee is aware of the organization (i.e., the source in our specific donation task).",
	PersuasionGame.S_TaskRelatedInquiry: "Please ask about the persuadee's opinion and expectation related to the task, such as their interests in knowing more about the organization.",
	PersuasionGame.S_PersonalRelatedInquiry: "Please ask about the persuadee's previous personal experiences relevant to charity donation.",
	PersuasionGame.S_Other: "Please respond naturally when no specific persuasion strategy applies.",
}


USER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.U_NoDonation: "The Persuadee declines or refuses to donate, or states they will not donate.",
	PersuasionGame.U_NegativeReaction: "The Persuadee reacts negatively, expresses doubt, or raises objections without clearly refusing.",
	PersuasionGame.U_Neutral: "The Persuadee remains undecided, neutral, or requests more information without showing clear sentiment.",
	PersuasionGame.U_PositiveReaction: "The Persuadee reacts positively or favorably but does not explicitly commit to donating.",
	PersuasionGame.U_Donate: "The Persuadee explicitly agrees or commits to donating.",
}


__all__ = [
	"SYSTEM_DIALOG_ACT_DEFINITIONS",
	"USER_DIALOG_ACT_DEFINITIONS",
]
