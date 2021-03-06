# frozen_string_literal: true

class SampleTree

  def initialize(sid)
    @sample = Sample
              .includes(:sample_type)
              .find(sid)
    @parents = []
  end

  def parents
    @sample
      .properties
      .select { |_k, v| v.class == Sample }
      .each_with_object({}) { |(k, sample), h| h[k] = SampleTree.new sample }
  end

  def expand
    @parents = parents
    self
  end

  def as_json

    samp = @sample.as_json(include: [
                             :sample_type,
                             items: { include: :object_type }
                           ])

    samp[:user_login] = @sample.user.login

    samp[:items].each do |i|

      i['data'] = JSON.parse i['data']
      if i['data']['from']
        i['data']['from'] = if i['data']['from'].class == String
                              [Item.find_by(id: i['data']['from']).as_json(include: :object_type)]
                            else
                              i['data']['from'].collect do |id|
                                Item.find_by(id: id).as_json(include: :object_type)
                              end
                            end
      end
    rescue StandardError
      i['data'] = {}

    end

    {
      sample: samp,
      parents: @parents.collect(&:as_json)
    }

  end

end
