import { RegDocument } from "@shared/api";

// Small seed corpus for practical experimentation.
export const seedCorpus: RegDocument[] = [
  {
    id: "md-mda-poultry-2021-01",
    title: "COMAR 15.11.11 Poultry Litter Management",
    agency: "Maryland Department of Agriculture",
    jurisdiction: "Maryland",
    year: 2021,
    sourceUrl: "https://dsd.maryland.gov/Pages/COMARHome.aspx",
    text:
      "(A) The purpose of this chapter is to establish requirements for the management of poultry litter to protect waters of the State. (B) Applicability. This chapter applies to persons who own or manage poultry operations generating or utilizing poultry litter. (C) Nutrient Management Plan. Operations shall implement and maintain a current nutrient management plan consistent with the Maryland Nutrient Management Manual. (D) Storage. Poultry litter shall be stored to prevent runoff, and uncovered field storage exceeding 90 days is prohibited near streams or wells. (E) Transportation Records. Producers shall retain transportation records for a minimum of 3 years, including quantity, dates, and receiving party. (F) Enforcement. The Department may issue notices of violation and assess civil penalties for noncompliance.",
  },
  {
    id: "md-mda-nutrient-2019-02",
    title: "Maryland Nutrient Management Manual — Phosphorus Management Tool",
    agency: "Maryland Department of Agriculture",
    jurisdiction: "Maryland",
    year: 2019,
    sourceUrl: "https://mda.maryland.gov/resource_conservation/Pages/nutrient_management.aspx",
    text:
      "This manual chapter describes the Phosphorus Management Tool (PMT) used to identify fields at risk for phosphorus loss and to guide nutrient application restrictions. Fields with high PMT scores require reduced or no application of phosphorus-bearing materials. Implementation is phased, with record-keeping requirements and periodic plan updates. Applicators must follow setbacks from surface waters and avoid application on frozen or snow-covered ground except as permitted.",
  },
  {
    id: "federal-epa-caa-2016-01",
    title: "EPA Clean Air Act General Provisions",
    agency: "U.S. Environmental Protection Agency",
    jurisdiction: "Federal",
    year: 2016,
    sourceUrl: "https://www.epa.gov/clean-air-act-overview",
    text:
      "The Clean Air Act establishes a comprehensive program for controlling air emissions from stationary and mobile sources. States are required to develop State Implementation Plans (SIPs). Agricultural operations may be subject to certain permitting and reporting requirements depending on pollutant thresholds and categories. Recordkeeping and monitoring provisions apply to regulated entities.",
  },
  {
    id: "md-mda-pesticide-2020-03",
    title: "Pesticide Applicator Certification and Recordkeeping",
    agency: "Maryland Department of Agriculture",
    jurisdiction: "Maryland",
    year: 2020,
    sourceUrl: "https://mda.maryland.gov/plants-pests/Pages/pesticide_regulation.aspx",
    text:
      "Commercial pesticide applicators shall be certified and maintain application records including date, location, pesticide product, EPA registration number, application rate, and weather conditions. Records shall be retained for a minimum of 2 years and made available upon request to the Department. Restricted use pesticides require additional supervision and documentation.",
  },
];
