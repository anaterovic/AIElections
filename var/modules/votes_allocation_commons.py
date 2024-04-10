from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

PARTY_NAMES = ["HDZ", "SDP", "MOST", "MOŽEMO", "DP"]

def visualize_mandate_allocation(votes_per_party: Dict[str, int], allocated_mandates: Dict[str, int], list_winners_num_votes: List[Tuple[str, int]]):
        num_rounds = sum(allocated_mandates.values())
        parties = ["HDZ", "SDP", "MOST", "DP", "MOŽEMO"]
        df = pd.DataFrame(index=parties, columns=["Total Votes"] + list(range(1, num_rounds+1)) + ["Seats Won", "True Seat Proportion"])
        for party in parties:
            df.loc[party, "Total Votes"] = round(votes_per_party[party])
        for round_index, (party, votes) in enumerate(list_winners_num_votes, start=1):
            df.loc[party, round_index] = round(votes)
        for col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        for party in parties:
            df.loc[party, "Seats Won"] = allocated_mandates.get(party, 0)
        total_votes_for_parties = sum(votes_per_party.values())
        true_proportion = {party: votes/total_votes_for_parties for party, votes in votes_per_party.items()}
        true_proportion = {party: round(votes*num_rounds, 2) for party, votes in true_proportion.items()}
        for party in parties:
            df.loc[party, "True Seat Proportion"] = true_proportion[party]
        # add a column which suggests a sorted list of which party was the closest to win the last seat and with which number of votes
        last_party_to_get_a_seat = list_winners_num_votes[-1][0]
        votes_per_party_before_last_seat_allocation = {}
        for party, votes in votes_per_party.items():
            if party == last_party_to_get_a_seat:
                number_to_divide_by = allocated_mandates.get(party, 0)
            else:
                number_to_divide_by = allocated_mandates.get(party, 0) + 1
            votes_per_party_before_last_seat_allocation[party] = round(votes/number_to_divide_by)
        closest_to_win_last_seat = sorted(votes_per_party_before_last_seat_allocation.items(), key=lambda x: x[1], reverse=True)
        for index, (party, virtual_votes) in enumerate(closest_to_win_last_seat, start=1):
            df.loc[party, "Votes to win last seat"] = str(virtual_votes) + " (" + str(index) + ")"
        return df


class BirackoMjesto:
    def __init__(self, name: str, fixed: List[Tuple[str, float]], data: pd.Series):
        self.name = name
        self.party_votes: Dict[str, int] = {
            "HDZ": data["HDZ"],
            "SDP": data["SDP"],
            "MOST": data["MOST"],
            "MOŽEMO": data["MOŽEMO"],
            "DP": data["DP"]
        }
        self.total: int = data["Glasovalo ukupno"]
        self.fixed_bool: Dict[str, bool] = self.__get_fixed_bool(fixed)
        self.fixed_percentages = self.__get_fixed_percentages(fixed)


    def __get_fixed_bool(self, fixed: List[Tuple[str, float]]) -> Dict[str, bool]:
        d = {party: False for party in PARTY_NAMES}
        for party, _ in fixed:
            d[party] = True
        return d
    
    def __get_fixed_percentages(self, fixed: List[Tuple[str, float]]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for party, percentage in fixed:
            d[party] = percentage
        return d
    
    def is_fixed(self, party: str) -> bool:
        return self.fixed_bool[party]
    
    def get_percentage(self, party: str, check_fixed: bool = True) -> float:
        if check_fixed and self.is_fixed(party):
            return self.fixed_percentages[party]
        return self.party_votes[party] / self.total
    
    def get_num_votes(self, party: str) -> int:
        return self.party_votes[party]


class IzbornaJedinica:
    def __init__(self, id: int, data: pd.DataFrame, fixed: Dict[str, List[Tuple[str, float]]]):
        self.id = id
        self.biracka_mjesta = self.__get_votes_for_parties_and_biracka_mjesta(data, fixed)


    def __get_votes_for_parties_and_biracka_mjesta(self, data: pd.DataFrame, fixed: Dict[str, List[Tuple[str, float]]]) -> List[BirackoMjesto]:
        biracka_mjesta: List[BirackoMjesto] = []
        for _, row in data.iterrows():
            name = row["BM"]
            biracka_mjesta.append(BirackoMjesto(name, fixed.get(name, []), row))
        return biracka_mjesta
    

    def get_gross_num_votes(self, party: str) -> int:
        num_votes = 0
        for bm in self.biracka_mjesta:
            num_votes += bm.get_num_votes(party)
        return num_votes
    
    def get_fixed_num_votes(self, party: str) -> int:
        num_votes = 0
        for bm in self.biracka_mjesta:
            if bm.is_fixed(party):
                num_votes += bm.get_num_votes(party)
        return num_votes


class IzbornaGodina:
    def __init__(self, year: int, data_izborne_jedinice: Dict[int, pd.DataFrame], fixed: Dict[str, List[Tuple[str, float]]]):
        self.year: int = year
        self.izborne_jedinice: List[IzbornaJedinica] = self.__get_votes_for_parties_and_izborne_jedinice(data_izborne_jedinice, fixed)


    def __get_votes_for_parties_and_izborne_jedinice(self, data_izborne_jedinice: Dict[int, pd.DataFrame], fixed: Dict[str, List[Tuple[str, float]]]) -> List[IzbornaJedinica]:
        izborne_jedinice: List[IzbornaJedinica] = []
        for id, data in data_izborne_jedinice.items():
            izborne_jedinice.append(IzbornaJedinica(id, data, fixed))
        return izborne_jedinice
    
    def get_gross_num_votes(self, party: str) -> int:
        num_votes = 0
        for ij in self.izborne_jedinice:
            num_votes += ij.get_gross_num_votes(party)
        return num_votes
    
    def get_fixed_num_votes(self, party: str) -> int:
        num_votes = 0
        for ij in self.izborne_jedinice:
            num_votes += ij.get_fixed_num_votes(party)
        return num_votes
    
    def get_variable_num_votes(self, party: str) -> int:
        return self.get_gross_num_votes(party) - self.get_fixed_num_votes(party)


class MandatePrediction:
    def __init__(self, posljednaIzbornaGodina: IzbornaGodina, var_predictions: Dict[str, float], num_voters: int, izlaznost_po_ij: Dict[int, float], broj_biraca_po_ij: Dict[int, int], ukljuci_izlaznost_po_ij: bool = True, ukljuci_izlaznost_kao_faktor: bool = False):
        self.posljednaIzbornaGodina = posljednaIzbornaGodina
        self.var_predictions = var_predictions
        self.num_voters = num_voters  # N
        self.izlaznost_po_ij = izlaznost_po_ij
        self.broj_biraca_po_ij = broj_biraca_po_ij
        self.ukljuci_izlaznost_po_ij = ukljuci_izlaznost_po_ij
        self.ukljuci_izlaznost_kao_faktor = ukljuci_izlaznost_kao_faktor

        # assert set(PARTY_NAMES).add("IZLAZNOST").issubset(var_predictions.keys()), f"Not all parties and IZLAZNOST are present in var_predictions. Keys in var_predictions: {var_predictions.keys()}"

        self.national_gross_votes_last_el: Dict[str, int] = {party: self.posljednaIzbornaGodina.get_gross_num_votes(party) for party in PARTY_NAMES}  # hdz
        self.national_fixed_votes_last_el: Dict[str, int] = {party: self.posljednaIzbornaGodina.get_fixed_num_votes(party) for party in PARTY_NAMES}  # f

        self.gross_votes_per_izborna_jedinica_last_el: Dict[int, Dict[str, int]] = {ij.id: {party: ij.get_gross_num_votes(party) for party in PARTY_NAMES} for ij in self.posljednaIzbornaGodina.izborne_jedinice}  # hdz1, hdz2, ...
        self.fixed_votes_per_izborna_jedinica_last_el: Dict[int, Dict[str, int]] = {ij.id: {party: ij.get_fixed_num_votes(party) for party in PARTY_NAMES} for ij in self.posljednaIzbornaGodina.izborne_jedinice}  # f1, f2, ...

    def get_gross_num_votes_pred(self, party: str, ij_id: Optional[int] = None) -> int:
        # HDZ
        if self.ukljuci_izlaznost_kao_faktor:
            izlaznost = self.var_predictions["IZLAZNOST"]
            base_izlaznost = 0.4685
            coefficient = base_izlaznost / izlaznost if party == "HDZ" else izlaznost / base_izlaznost if party == "SDP" else 1.0
            for _ in range(2):
                coefficient = (1.0 + coefficient) / 2
        else:
            coefficient = 1.0
        
        if not ij_id:
            return int(self.num_voters * self.var_predictions["IZLAZNOST"] * self.var_predictions[party] * coefficient)
        return int(self.broj_biraca_po_ij[ij_id] * self.izlaznost_po_ij[ij_id] * self.var_predictions[party] * coefficient)
    
    def get_variable_votes_last_el(self, ij_id: Union[int, None], party: str) -> int:
        # hdz' ili hdz1', hdz2', ...
        if not ij_id:
            return self.national_gross_votes_last_el[party] - self.national_fixed_votes_last_el[party]
        return self.gross_votes_per_izborna_jedinica_last_el[ij_id][party] - self.fixed_votes_per_izborna_jedinica_last_el[ij_id][party]
    
    def estimate_num_variable_votes_received(self, party: str, ij_id: Optional[int] = None) -> int:
        # HDZ'
        if not ij_id:
            return self.get_gross_num_votes_pred(party=party) - self.national_fixed_votes_last_el[party]
        return self.get_gross_num_votes_pred(party=party, ij_id=ij_id) - self.fixed_votes_per_izborna_jedinica_last_el[ij_id][party]

    def predict_gross_num_votes(self, ij_id: int, party: str) -> int:
        # HDZ1, HDZ2, ...
        ratio_of_variable_votes_in_izborna_jedinica: float = self.get_variable_votes_last_el(ij_id, party) / self.get_variable_votes_last_el(None, party)

        if self.ukljuci_izlaznost_po_ij:
            return self.fixed_votes_per_izborna_jedinica_last_el[ij_id][party] + self.estimate_num_variable_votes_received(party=party, ij_id=ij_id)
        return self.fixed_votes_per_izborna_jedinica_last_el[ij_id][party] + self.estimate_num_variable_votes_received(party=party) * ratio_of_variable_votes_in_izborna_jedinica
    
    def dhondt(self, votes: Dict[str, int], mandates: int) -> Tuple[Dict[str, int], List[Tuple[str, int]]]:
        allocated_mandates: Dict[str, int] = {}
        list_winners_num_votes: List[Tuple[str, int]] = []
        parties = list(votes.keys())
        for _ in range(mandates):
            max_votes = 0
            max_party = ""
            for party in parties:
                party_votes = votes[party]
                quotient = party_votes / (allocated_mandates.get(party, 0) + 1)
                if quotient > max_votes:
                    max_votes = quotient
                    max_party = party
            allocated_mandates[max_party] = allocated_mandates.get(max_party, 0) + 1
            list_winners_num_votes.append((max_party, round(max_votes)))
        return {"allocated_mandates": allocated_mandates, "list_winners_num_votes": list_winners_num_votes}
    
    def get_mandate_allocation_for_izborna_jedinica(self, ij_id: int, mandates: int) -> Dict[str, Any]:
        votes = {party: self.predict_gross_num_votes(ij_id, party) for party in PARTY_NAMES}
        d = self.dhondt(votes, mandates)
        d["votes_per_party"] = votes
        return d
    
    def get_mandate_allocation(self, mandates: int) -> Dict[int, Dict[str, Any]]:
        return {ij.id: self.get_mandate_allocation_for_izborna_jedinica(ij.id, mandates) for ij in self.posljednaIzbornaGodina.izborne_jedinice}
